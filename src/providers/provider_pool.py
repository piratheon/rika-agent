"""Provider pool — singleton, multi-provider failover with correct key rotation."""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

from src.db import key_store
from src.providers.base_provider import ProviderAuthError, ProviderQuotaError, ProviderTransientError
# imported lazily inside methods to avoid circular import:
#   from src.providers.groq_provider import GroqToolUseFailedError
from src.utils.logger import logger

_VIRTUAL_KEY_USAGE: Dict[str, datetime] = {}
_MAX_TRANSIENT_PER_KEY = 3
_KEYLESS_PROVIDERS = frozenset({"ollama", "g4f"})

_pool_instance: Optional["ProviderPool"] = None

def get_pool() -> "ProviderPool":
    global _pool_instance
    if _pool_instance is None:
        _pool_instance = ProviderPool()
    return _pool_instance

class ProviderPool:
    # Per-provider tool count caps. Groq's llama models fail with >8-10 tools.
    _TOOL_CAPS: Dict[str, int] = {
        "groq": 6,
    }

    def __init__(self) -> None:
        self._locks: Dict[tuple, asyncio.Lock] = {}
        self._working_caps: Dict[str, int] = {}
        self._logged_env_check: Set[int] = set()

    def _get_lock(self, user_id: int, provider: str) -> asyncio.Lock:
        key = (user_id, self._normalize(provider))
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    async def request_with_key(self, user_id: int, provider: str, payload: dict) -> dict:
        logger.info("provider_pool_request_start", user_id=user_id, provider=provider)
        
        # Try the requested provider first
        try:
            return await self._request_with_key_impl(user_id, provider, payload)
        except Exception as exc:
            logger.warning("provider_pool_primary_failed", provider=provider, error=str(exc))
            
            # If g4f is enabled, try it as final fallback
            if self._is_g4f_enabled():
                logger.info("provider_pool_trying_g4f_fallback", user_id=user_id)
                try:
                    return await self._request_with_key_impl(user_id, "g4f", payload)
                except Exception as g4f_exc:
                    logger.error("provider_pool_g4f_fallback_failed", error=str(g4f_exc))
            
            # All providers failed
            raise RuntimeError(f"All providers failed. Last error: {str(exc)}")
    
    def _is_g4f_enabled(self) -> bool:
        """Check if G4F is enabled in config."""
        try:
            from src.config import Config
            cfg = Config.get()
            return getattr(cfg, "g4f_enabled", False)
        except Exception:
            return False
    
    async def _request_with_key_impl(self, user_id: int, provider: str, payload: dict) -> dict:
        """Internal implementation - try a single provider."""
        norm = self._normalize(provider)
        if norm in _KEYLESS_PROVIDERS:
            logger.info("provider_pool_keyless", provider=norm)
            adapter = self._make_adapter(norm, "")
            try:
                return await adapter.request(payload)
            except (ProviderAuthError, ProviderQuotaError, ProviderTransientError):
                raise
            except Exception as exc:
                raise ProviderTransientError(str(exc))

        tried: Set[str] = set()
        transient_streak = 0
        rate_limit_retries = 0
        max_rate_limit_retries = 3
        rate_limit_delay = 3

        while True:
            # Acquire lock with timeout to prevent deadlocks
            lock = self._get_lock(user_id, norm)
            logger.info("provider_pool_acquiring_lock", user_id=user_id, provider=norm)
            try:
                await asyncio.wait_for(lock.acquire(), timeout=5.0)
                logger.info("provider_pool_lock_acquired", user_id=user_id, provider=norm)
            except asyncio.TimeoutError:
                logger.error("provider_pool_lock_timeout", user_id=user_id, provider=norm)
                raise RuntimeError(f"Lock timeout for {provider}")
            
            try:
                logger.info("provider_pool_selecting_key", user_id=user_id, provider=norm)
                k = await self._select_key(user_id, norm, exclude=tried)
                logger.info("provider_pool_key_selected", user_id=user_id, provider=norm, key_id=k["id"] if k else None)
            finally:
                lock.release()

            if k is None:
                logger.error("provider_pool_no_key", user_id=user_id, provider=norm)
                raise RuntimeError(f"No {provider} API key found. Add one with /addkey {provider}:\"key\"")

            api_key: str = k["raw_key"]
            logger.info("provider_pool_got_key", user_id=user_id, provider=norm)
            adapter = self._make_adapter(norm, api_key)

            try:
                logger.info("provider_pool_making_request", user_id=user_id, provider=norm)
                resp = await adapter.request(payload)
                logger.info("provider_pool_request_success", user_id=user_id, provider=norm)
                await self._record_usage(k)
                # Token accounting
                usage = resp.get("usage")
                if usage and k["id"] >= 0:
                    tokens = usage.get("total_tokens") or usage.get("total_token_count", 0)
                    if tokens:
                        try:
                            await key_store.increment_tokens_used(k["id"], int(tokens))
                        except Exception:
                            pass
                return resp

            except ProviderAuthError as exc:
                logger.error("provider_pool_auth_failed", key_id=k["id"], provider=norm, error=str(exc))
                if k["id"] >= 0:
                    await key_store.blacklist_key(k["id"], reason="auth_failed")
                tried.add(api_key)
                transient_streak = 0
                rate_limit_retries = 0

            except ProviderQuotaError as exc:
                err_lower = str(exc).lower()
                is_rate_limit = any(x in err_lower for x in ["429","rate limit","too many","tpm","rpm"])
                if is_rate_limit:
                    rate_limit_retries += 1
                    logger.warning("provider_pool_rate_limit", provider=norm, key_id=k["id"], retry=rate_limit_retries, max=max_rate_limit_retries)

                    if rate_limit_retries < max_rate_limit_retries:
                        # Wait and retry same key
                        logger.info("provider_pool_rate_limit_retrying", provider=norm, delay=rate_limit_delay)
                        await asyncio.sleep(rate_limit_delay)
                        continue
                    else:
                        # Max retries reached, try next provider
                        logger.warning("provider_pool_rate_limit_max_retries", provider=norm, retries=rate_limit_retries)
                        tried.add(api_key)
                        rate_limit_retries = 0
                        # Raise to trigger fallback to next provider
                        raise ProviderQuotaError(f"Rate limit exceeded after {max_rate_limit_retries} retries")
                else:
                    logger.warning("provider_pool_hard_quota", provider=norm, key_id=k["id"])
                    if k["id"] >= 0:
                        await key_store.blacklist_key(k["id"], reason="quota_exceeded",
                                                      quota_resets_at=self._estimate_reset(norm))
                    tried.add(api_key)
                    transient_streak = 0
                    rate_limit_retries = 0

            except (ProviderTransientError, Exception) as exc:
                transient_streak += 1
                logger.warning("provider_pool_transient_error", provider=norm, key_id=k["id"],
                               streak=transient_streak, error=str(exc))
                if transient_streak >= _MAX_TRANSIENT_PER_KEY:
                    tried.add(api_key)
                    transient_streak = 0
                    rate_limit_retries = 0
                else:
                    await asyncio.sleep(min(2 ** transient_streak, 30))

    async def stream_with_key(self, user_id: int, provider: str, payload: dict):
        norm = self._normalize(provider)
        if norm in _KEYLESS_PROVIDERS:
            async for chunk in self._make_adapter(norm, "").stream(payload):
                yield chunk
            return
        k = await self._select_key(user_id, norm)
        if k is None:
            raise RuntimeError(f"No key available for {provider}")
        adapter = self._make_adapter(norm, k["raw_key"])
        try:
            async for chunk in adapter.stream(payload):
                yield chunk
            await self._record_usage(k)
        except (ProviderQuotaError, ProviderAuthError) as exc:
            reason = "auth_failed" if isinstance(exc, ProviderAuthError) else "quota_exceeded"
            if k["id"] >= 0:
                await key_store.blacklist_key(k["id"], reason=reason)
            raise

    async def get_healthy_key(self, user_id: int, provider: str) -> Optional[dict]:
        norm = self._normalize(provider)
        if norm in _KEYLESS_PROVIDERS:
            adapter = self._make_adapter(norm, "")
            try:
                ok = await adapter.test_key()
                return {"id": -99, "provider": norm, "raw_key": "", "is_blacklisted": False} if ok else None
            except Exception:
                return None
        k = await self._select_key(user_id, norm)
        if k is None:
            return None
        adapter = self._make_adapter(norm, k["raw_key"])
        try:
            if await adapter.test_key():
                return k
        except (ProviderAuthError, ProviderQuotaError) as exc:
            reason = "auth_failed" if isinstance(exc, ProviderAuthError) else "quota_exceeded"
            if k["id"] >= 0:
                await key_store.blacklist_key(k["id"], reason=reason)
        except Exception:
            pass
        return None


    async def request_with_key_structured(self, user_id: int, provider: str,
                                           payload: dict, tool_schemas: list) -> "StructuredResponse":
        """Like request_with_key but returns a StructuredResponse for function calling.

        Has the same key rotation and blacklisting logic as request_with_key.
        Additionally handles GroqToolUseFailedError by retrying with fewer tools
        instead of burning the key (the key is valid — the schema was too large).
        """
        from src.providers.base_provider import ProviderAuthError, ProviderQuotaError, StructuredResponse
        from src.providers.groq_provider import GroqToolUseFailedError
        norm = self._normalize(provider)
        if norm in _KEYLESS_PROVIDERS:
            adapter = self._make_adapter(norm, "")
            return await adapter.request_with_tools(payload, tool_schemas)

        tried: Set[str] = set()
        tool_use_failures = 0
        active_schemas = self._cap_tools(norm, tool_schemas)

        for _attempt in range(5):
            k = await self._select_key(user_id, norm, exclude=tried)
            if k is None:
                raise RuntimeError(
                    f"No {provider} API key available. "
                    f"Add one with /addkey {provider}:\"key\""
                )
            adapter = self._make_adapter(norm, k["raw_key"])
            try:
                resp = await adapter.request_with_tools(payload, active_schemas)
                await self._record_usage(k)
                if norm in self._TOOL_CAPS:
                    self._working_caps[norm] = len(active_schemas)
                return resp
            except GroqToolUseFailedError:
                tool_use_failures += 1
                logger.warning(
                    "structured_tool_use_failed",
                    provider=norm,
                    attempt=tool_use_failures,
                    schemas_before=len(active_schemas),
                )
                if tool_use_failures >= 2:
                    raise RuntimeError(
                        f"{provider} tool_use_failed after {tool_use_failures} retries"
                    )
                active_schemas = self._cap_tools(norm, active_schemas, reduced=True)
                await asyncio.sleep(0.5)
                continue
            except ProviderAuthError as exc:
                logger.error("structured_auth_failed", provider=norm, key_id=k["id"], error=str(exc))
                if k["id"] >= 0:
                    await key_store.blacklist_key(k["id"], reason="auth_failed")
                tried.add(k["raw_key"])
            except ProviderQuotaError as exc:
                err_lower = str(exc).lower()
                is_tpm = any(x in err_lower for x in ["tpm", "tokens per minute", "rpm", "requests per minute"])
                if is_tpm:
                    # TPM/RPM resets in seconds — sleep briefly and retry the SAME key
                    wait = 15
                    logger.warning("structured_tpm_limit", provider=norm, key_id=k["id"],
                                   wait_seconds=wait, error=str(exc)[:120])
                    await asyncio.sleep(wait)
                    # Do NOT add to tried — key is still valid
                else:
                    logger.warning("structured_quota_exceeded", provider=norm, key_id=k["id"], error=str(exc)[:120])
                    if k["id"] >= 0:
                        await key_store.blacklist_key(
                            k["id"], reason="quota_exceeded",
                            quota_resets_at=self._estimate_reset(norm),
                        )
                    tried.add(k["raw_key"])
            except Exception as exc:
                logger.warning("structured_transient_error", provider=norm, error=str(exc))
                tried.add(k["raw_key"])
                await asyncio.sleep(1)

        raise RuntimeError(f"All {provider} keys exhausted for structured request.")

    async def _select_key(self, user_id: int, provider: str, exclude: Optional[Set[str]] = None) -> Optional[dict]:
        exclude = exclude or set()
        db_keys = await key_store.list_api_keys(user_id)
        norm_p = self._normalize(provider)
        provider_keys = [k for k in db_keys if self._normalize(k.get("provider", "")) == norm_p]
        
        # Check env vars - try both normalized and original provider names
        env_variants = [
            f"{provider.upper()}_API_KEY",
            f"{norm_p.upper()}_API_KEY",
        ]
        # Special case for gemini/google
        if norm_p == "gemini":
            env_variants.append("GOOGLE_API_KEY")
        
        for env_var in env_variants:
            env_str = os.environ.get(env_var, "")
            if env_str.strip():
                logger.info("provider_pool_found_env_key", provider=provider, env_var=env_var)
                break
        
        for i, raw in enumerate([r.strip() for r in env_str.replace(",", " ").split() if r.strip()]):
            usage_key = f"{provider}:{raw[:12]}"
            provider_keys.append({"id": -(i+1), "provider": provider, "raw_key": raw,
                                  "is_blacklisted": False,
                                  "last_used_at": _VIRTUAL_KEY_USAGE.get(usage_key, datetime.min).isoformat(),
                                  "quota_resets_at": None, "usage_key": usage_key})
        if not provider_keys:
            logger.warning("provider_pool_no_keys_found", provider=provider, user_id=user_id, 
                          env_vars_checked=env_variants, db_keys_count=len(db_keys))
            return None

        def _lru(k):
            try:
                return datetime.fromisoformat(k.get("last_used_at") or "1970-01-01T00:00:00")
            except ValueError:
                return datetime.min

        for k in sorted(provider_keys, key=_lru):
            if k.get("is_blacklisted"):
                reset = k.get("quota_resets_at")
                if reset:
                    try:
                        if datetime.fromisoformat(reset) <= datetime.utcnow():
                            if k["id"] >= 0:
                                await key_store.unblacklist_key(k["id"])
                            k["is_blacklisted"] = False
                        else:
                            continue
                    except ValueError:
                        continue
                else:
                    continue
            if k["id"] >= 0 and "raw_key" not in k:
                try:
                    k["raw_key"] = (await key_store.get_api_key_raw(k["id"])).decode("utf-8")
                except Exception:
                    if k["id"] >= 0:
                        await key_store.blacklist_key(k["id"], reason="decryption_failed")
                    continue
            if k.get("raw_key", "") in exclude:
                continue
            return k
        return None

    async def _record_usage(self, k: dict) -> None:
        if k["id"] >= 0:
            try:
                await key_store.update_key_last_used(k["id"])
            except Exception:
                pass
        elif k.get("usage_key"):
            _VIRTUAL_KEY_USAGE[k["usage_key"]] = datetime.utcnow()

    def _normalize(self, name: str) -> str:
        n = (name or "").lower().strip()
        return "gemini" if n in ("google", "gemini") else n

    def _cap_tools(self, provider: str, tool_schemas: list, reduced: bool = False) -> list:
        """Enforce per-provider tool count cap.

        Args:
            reduced: If True, halve the cap (used on tool_use_failed retry).
        """
        cap = self._TOOL_CAPS.get(provider)
        if cap is None:
            return tool_schemas
        working = self._working_caps.get(provider, cap)
        effective_base = min(cap, working)
        effective = effective_base // 2 if reduced else effective_base
        if len(tool_schemas) <= effective:
            return tool_schemas
        logger.info(
            "provider_pool_tool_cap_applied",
            provider=provider,
            original=len(tool_schemas),
            capped=effective,
        )
        # Prefer keeping declare_step first, then the most broadly useful tools
        priority = ["declare_step", "web_search", "run_shell_command", "run_python",
                    "curl", "read_file", "write_file", "delegate_task"]
        schema_map = {s.name: s for s in tool_schemas if hasattr(s, 'name')}
        ordered = [schema_map[n] for n in priority if n in schema_map]
        remaining = [s for s in tool_schemas if s not in ordered]
        return (ordered + remaining)[:effective]

    def _make_adapter(self, provider: str, api_key: str):
        norm = self._normalize(provider)
        if norm == "gemini":
            from src.providers.gemini_provider import GeminiProvider
            return GeminiProvider(api_key)
        if norm == "openrouter":
            from src.providers.openrouter_provider import OpenRouterProvider
            return OpenRouterProvider(api_key)
        if norm == "groq":
            from src.providers.groq_provider import GroqProvider
            return GroqProvider(api_key)
        if norm == "ollama":
            from src.providers.ollama_provider import OllamaProvider
            return OllamaProvider()
        if norm == "g4f":
            from src.providers.g4f_provider import G4FProvider
            return G4FProvider()
        from src.providers.openrouter_provider import OpenRouterProvider
        return OpenRouterProvider(api_key)

    async def get_available_providers(self, user_id: int) -> List[str]:
        """Return list of providers that have valid API keys available."""
        available = []
        db_keys = await key_store.list_api_keys(user_id)
        
        # Check common providers
        for provider in ["gemini", "groq", "openrouter", "ollama", "g4f"]:
            # Check env keys
            env_key = os.environ.get(f"{provider.upper()}_API_KEY", "")
            if env_key.strip():
                available.append(provider)
                continue
            
            # Check DB keys (non-blacklisted)
            provider_keys = [k for k in db_keys 
                           if self._normalize(k.get("provider", "")) == provider 
                           and not k.get("is_blacklisted")]
            if provider_keys:
                available.append(provider)
        
        return available

    def _estimate_reset(self, provider: str) -> Optional[str]:
        try:
            from src.config import Config
            cfg = Config.get()
            field = f"{provider}_quota_reset_utc_hour"
            if hasattr(cfg, field):
                hour = int(getattr(cfg, field))
                now = datetime.utcnow()
                candidate = now.replace(hour=hour, minute=0, second=0, microsecond=0)
                if candidate <= now:
                    candidate += timedelta(days=1)
                return candidate.isoformat()
        except Exception:
            pass
        return None
