<div align="center">

<br/>

```
██████╗ ██╗██╗  ██╗ █████╗       █████╗  ██████╗ ███████╗███╗   ██╗████████╗
██╔══██╗██║██║ ██╔╝██╔══██╗     ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
██████╔╝██║█████╔╝ ███████║     ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   
██╔══██╗██║██╔═██╗ ██╔══██║     ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   
██║  ██║██║██║  ██╗██║  ██║     ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   
╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝  ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   
```

<h3>Self-hosted agentic AI on your own hardware — Telegram-native, privacy-first.</h3>


<br/>

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Telegram](https://img.shields.io/badge/Telegram-Bot_API-26A5E4?style=flat-square&logo=telegram&logoColor=white)](https://core.telegram.org/bots)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue?style=flat-square)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)](docker-compose.yml)
[![SQLite](https://img.shields.io/badge/Storage-SQLite_%2F_Postgres-003B57?style=flat-square&logo=sqlite&logoColor=white)](#architecture)
[![Security](https://img.shields.io/badge/Keys-AES--256--GCM-dc2626?style=flat-square&logo=keepassxc&logoColor=white)](#security)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/piratheon/rika-agent)

**Online Providers:**

[![NVIDIA](https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=flat-square&logo=nVIDIA&logoColor=white)](#providers)
[![Google GenAI](https://img.shields.io/badge/Google%20GenAI-886FBF?logo=googlegemini&logoColor=fff)](#providers)
[![Groq](https://img.shields.io/badge/GROQ-f97316?style=flat-square)](#providers)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-7c3aed?style=flat-square)](#providers)

<br/>

</div>

---
> **rika-agent** is a privacy first open source agentic system that you can control via Telegram (using a bot) and soon via other interfaces.
 
>v2.1.9 ships JSON-native function calling, provider failover with live countdown UX, a three-level code sandbox, background system monitoring, vision input, file delivery, encrypted memory, and support for Groq / Gemini / OpenRouter / NVIDIA NIM / Vercel AI / Ollama / G4F — running entirely on your own server, with your data staying yours.

---

## What it actually does

Most "AI assistants" are wrappers that relay your messages to an API. rika-agent is different: it runs as a persistent process on your machine with real access to your shell, file system, and network. The agent decides when to use tools, chains multiple calls together, monitors your server in the background, and delivers results — including actual files — back to you in Telegram.

```
you → "analyze the last 200 nginx errors, find patterns, write a report"

rika → [run_shell_command: tail -200 /var/log/nginx/error.log]
        [web_search: common nginx 502 causes 2025]
        [run_python: parse log entries, cluster by error type]
        [writes report.md to workspace]
        [sends report.md as Telegram file attachment]
        "Found 3 patterns. Most common: upstream timeout (67%). Report attached."
```

No copy-paste. No switching tabs. It just does it.

---

## Features

<table>
<tr>
<td width="50%">

**Core agent**
- JSON function calling via native provider APIs — no text protocol
- Multi-turn ReAct loop (up to 20 tool calls per request)
- Mandatory `end_thinking` exit — responses are never delivered mid-thought
- Automatic correction loop when LLM forgets to use a tool call
- 3-tier complexity classifier (skips orchestration for simple queries)
- Per-user concurrency limit (max 2 simultaneous tasks)

</td>
<td width="50%">

**Providers**
- Gemini, Groq, OpenRouter, NVIDIA NIM, Vercel AI — key rotation with LRU selection
- Ollama — local models, zero cost, zero data egress
- G4F — free endpoints (GPT-4o, Llama), no key needed
- Live 30s countdown with **Stop / Retry now** buttons when all providers fail
- Blacklist + quota-reset scheduling per provider
- Per-provider schema cap and tool-count floor so function calling never degrades to zero

</td>
</tr>
<tr>
<td>

**Background monitoring** *(zero AI tokens at rest)*
- `SystemWatcher` — CPU load, memory %, disk % via `/proc`
- `ProcessWatcher` — process presence detection
- `URLWatcher` — HTTP health check, state-change alerts
- `PortWatcher` — TCP port availability
- `LogPatternWatcher` — regex tail on any log file
- `CronWatcher` — scheduled AI tasks on any interval
- `ScriptWatcher` — execute agent-authored monitoring scripts
- AI-driven setup: `/autowatch "guard my server"` → auto-creates watchers
- One LLM call per anomaly → Telegram alert
- All configs survive restarts

</td>
<td>

**Tools**
- `web_search` — DuckDuckGo, no API key, bot-detection fallback
- `wikipedia_search` — MediaWiki REST API
- `curl` — fetch any URL, stripped to readable text
- `run_shell_command` — host shell, cwd = workspace
- `run_python` — configurable isolation sandbox
- `send_file` — deliver workspace files to Telegram
- `list_workspace` — browse the agent's sandbox
- `save_memory` / `get_memories` / `save_skill` — per-user, isolated
- `delegate_task` — spawn a research sub-agent

</td>
</tr>
<tr>
<td>

**Security**
- **Owner-only auth gate** — `OWNER_USER_ID` in `.env` limits the bot to one Telegram user. All other messages are silently dropped. Implemented as a PTB TypeHandler at group=-1: zero per-handler code.
- AES-256-GCM encryption for all stored API keys
- 22-rule shell command firewall — pure Python, zero AI tokens
  - **CRITICAL**: `rm -rf /`, fork bombs, disk wipes → blocked unconditionally
  - **HIGH**: kill init, flush iptables → blocked in standard/strict
  - **MEDIUM**: `curl | bash`, `sudo su` → `CONFIRM:` prefix to override
  - Secondary structural block: `$()`, backticks, `;bash/curl/rm` chains → always blocked
- File delivery path-traversal protection (`Path.resolve()` + `is_relative_to()`)

</td>
<td>

**Memory & context**
- Qdrant vector store — semantic recall across sessions
- Per-user key-value memory (facts, preferences, skills) — fully isolated
- Token-efficient injection: pinned (max 5) + relevant (top 4) only
- Memory pinning: `/pinmemory` / `/unpinmemory` commands
- Auto-summarization when context window fills
- Runtime context injected per-message (time, host, OS, user)
- Config singleton with 30s TTL — no disk read per message

</td>
</tr>
<tr>
<td>

**Vision**
- Send any photo → bot downloads, base64-encodes, sends to provider
- Gemini (native multimodal), OpenRouter (vision models), NVIDIA NIM
- Caption becomes the query; no caption = full description

</td>
<td>

**UX**
- Live countdown with **⏹ Stop** and **↩ Retry now** buttons during provider failures
- `/stop` command and Stop button reliably cancel the active asyncio task
- LiveBubble™ — throttled Telegram message edits with Braille spinner
- `AGENT_NAME=lain` in `.env` — name your agent anything you want
- `/reload` hot-reloads config + registry without restart
- Command audit log via `/cmdhistory`
- Workspace management via `/files`, `/cleanworkspace`

</td>
</tr>
</table>

---

## Quick start

**Requirements:** Python 3.12+, Telegram bot token from [@BotFather](https://t.me/BotFather)

```bash
git clone https://github.com/piratheon/rika-agent
cd rika-agent
bash scripts/bot_setup.sh
```

The setup wizard handles everything: bot token, owner Telegram ID, agent name, provider keys, sandbox level detection, optional Ollama/G4F/NVIDIA/Vercel configuration, and database migration. When done:

```bash
bash scripts/start.sh
```

**Docker:**
```bash
cp .env.template .env   # fill TELEGRAM_BOT_TOKEN, BOT_ENCRYPTION_KEY, OWNER_USER_ID
docker compose up -d
docker compose logs -f
```

---

## Configuration

### `.env` — secrets

```env
# Required
TELEGRAM_BOT_TOKEN=
BOT_ENCRYPTION_KEY=   # generate: python3 -c "import secrets; print(secrets.token_hex(32))"

# Security — strongly recommended for any non-local deployment
OWNER_USER_ID=        # your Telegram user ID (message @userinfobot to find it)
                      # if unset, the bot responds to ALL Telegram users

# Optional
DATABASE_PATH=./data/rk.db
AGENT_NAME=rika       # display name — purely cosmetic

# Provider keys (can also be added later with /addkey in Telegram)
GEMINI_API_KEY=
GROQ_API_KEY=
OPENROUTER_API_KEY=
NVIDIA_API_KEY=       # auto-enables NVIDIA NIM when set (nvapi-...)
VERCEL_API_KEY=       # Vercel AI Gateway — also set VERCEL=1
VERCEL=1

# Optional: Postgres backend (SQLite is used if unset)
POSTGRES_URL=         # postgresql://user:pass@host:5432/db
                      # pip install asyncpg

# Optional: Vercel KV / Upstash Redis
KV_REST_API_URL=
KV_REST_API_TOKEN=
```

> **`BOT_ENCRYPTION_KEY` is critical.** All stored API keys are encrypted with it. If you lose it, stored keys are unrecoverable. Back it up separately from the database.

### `config.json` — behavior

```json
{
  "bot_name": "rika-agent",
  "default_model": "gemini-2.0-flash",
  "default_provider_priority": ["groq", "openrouter", "gemini"],

  "sandbox_level": 1,
  "enable_command_security": true,
  "command_security_level": "standard",

  "workspace_path": "~/.rika/shared",
  "workspace_max_size_mb": 500,

  "provider_max_retries": 2,
  "provider_retry_delay": 2.0,

  "ollama_enabled": false,
  "ollama_base_url": "http://localhost:11434",
  "ollama_default_model": "llama3.2",

  "g4f_enabled": false,
  "g4f_model": "MiniMaxAI/MiniMax-M2.5",

  "nvidia_enabled": false,
  "nvidia_model": "z-ai/glm-5.1",
  "nvidia_auto_detect": true,

  "vercel_enabled": false,
  "vercel_model": "openai/gpt-4o-mini",
  "vercel_auto_detect": true,

  "max_context_messages": 40,
  "max_concurrent_orchestrations_per_user": 2,
  "max_background_agents_per_user": 10,
  "tool_timeout_seconds": 10,

  "groq_model": "llama-3.3-70b-versatile",
  "openrouter_model": "google/gemini-2.0-flash-001",
  "gemini_model": "gemini-2.0-flash",
  "ollama_model": "llama3.2"
}
```

> Existing `config.json` files work without modification. All new fields have safe defaults.

### `soul.md` — personality

The agent's tone, instructions, and character. Loaded at startup. Gitignored by default. Changes take effect on `/reload` or within 30 seconds.

```bash
cp soul.md.template soul.md
$EDITOR soul.md
```

---

## Providers

### Keyed

| Provider | Free tier | Notes |
|---|---|---|
| **Gemini** | Yes (generous) | Best multimodal, 1M context, native vision |
| **Groq** | Yes | Fastest inference — llama3.3-70b, mixtral |
| **OpenRouter** | Pay-per-token | 200+ models including GPT-4o, Claude 3.5 |
| **NVIDIA NIM** | Free credits | nemotron-4-340b, llama-3.1-70b, mixtral-8x7b, gemma-7b |
| **Vercel AI** | Pay-per-token | AI Gateway with automatic provider routing |

Add any key via `/addkey` in Telegram:
```
/addkey groq:"gsk_..."
/addkey gemini:"AIza..."
/addkey nvidia:"nvapi-..."
/addkey openrouter:"sk-or-..."
/addkey vercel:"..."
```

Multiple keys per provider — the pool rotates automatically on rate limits or quota exhaustion. If all providers fail, rika shows a 30-second live countdown with Stop and Retry now buttons rather than giving up.

### Ollama & G4F

<details>
<summary><b>Ollama — local inference, zero cost, data never leaves your machine</b></summary>

```bash
# Install: https://ollama.com
ollama pull llama3.2
ollama serve
```

```json
{
  "ollama_enabled": true,
  "ollama_base_url": "http://localhost:11434",
  "ollama_default_model": "llama3.2",
  "default_provider_priority": ["ollama", "groq", "gemini"]
}
```

Auto-discovers models via `/api/tags`. Falls back to first available if requested model not found.

</details>

<details>
<summary><b>G4F — free access to GPT-4o, Claude, Gemini Pro</b></summary>

> **Warning:** G4F relies on reverse-engineered endpoints. It can break at any time. Use as last-resort fallback only.

```bash
pip install g4f
```

```json
{ "g4f_enabled": true }
```

</details>

---

## Code sandbox

Set `sandbox_level` in `config.json`:

| Level | Name | What the agent can do | Requirements |
|---|---|---|---|
| `0` | RestrictedPython | Arithmetic and logic only. No file I/O, no imports. Auto-escalates to Level 1 when code contains import statements. | None |
| `1` | Process + ulimit | Full Python, installed packages, write to workspace. CPU/RAM capped. | Linux / macOS |
| `2` | Docker | No network, memory-capped, ephemeral container. Maximum isolation. | Docker running |

The setup wizard detects your environment and recommends the highest available level.

> Level 2 is strongly recommended for any multi-user or public deployment.

---

## Background monitoring

```
/watch system  (every 120s)
      │
      └─ SystemWatcher.check()     ← pure Python, reads /proc, zero API cost
              │  load > 4.0
              ▼
          WakeSignal → queue
              │
              └─ WakeProcessor
                      └─ 1 LLM call → Telegram alert

"[CRIT] sys_3f7a — Load hit 6.2, memory at 91%.
 Likely runaway process. Check: ps aux --sort=-%mem | head -10"
```

```
/watch system                               CPU / memory / disk
/watch system cpu:90 mem:95                 custom thresholds
/watch process postgres                     process presence
/watch url https://mysite.com               HTTP health check
/watch port 5432                            TCP availability
/watch log /var/log/app.log "ERROR|FATAL"   regex in log file
/watch cron 30m summarize disk and warn if above 80%
/autowatch "guard my server"                AI-driven watcher setup

/watchers          list active agents
/stopwatch <id>    stop one
/wakelog           recent alerts
```

---

## Command reference

| Command | Description |
|---|---|
| `/start` | Initialize the bot |
| `/help` | Full command list |
| `/addkey provider:"key"` | Add an API key (groq, gemini, openrouter, nvidia, vercel, ollama) |
| `/status` | Keys, model, active agents, vector store state |
| `/providers` | Provider connectivity + Ollama model list |
| `/stop` | Cancel the currently running task or countdown |
| `/reload` | Hot-reload config (owner only) |
| `/broadcast` | Send a message to all users (owner only) |
| `/memory` | List stored memories and skills |
| `/pinmemory <key>` | Pin a memory for always-injection (max 5) |
| `/unpinmemory <key>` | Remove from always-injected list |
| `/deletememory <key>` | Delete a memory entry |
| `/autowatch <goal>` | AI-driven watcher setup (natural language) |
| `/watch <type> ...` | Register background monitor |
| `/watchers` | List active monitors |
| `/stopwatch <id>` | Stop a monitor |
| `/wakelog` | Recent wake events |
| `/files` | Workspace tree listing |
| `/cleanworkspace` | Wipe workspace contents |
| `/cmdhistory` | Command audit log |
| `/delete_me` | Delete all your data |

---

## Architecture

```
Telegram (app.py)
├── _auth_gate ─────────► PTB TypeHandler group=-1 — rejects non-owner silently
├── Photo ──────────────► vision provider → reply
├── Simple ─────────────► ProviderPool.request_with_key() → reply
└── Complex ────────────► orchestration loop
                               while turn < max_turns (20):
                                 request_with_key_structured() → StructuredResponse
                                 if raw text → inject correction, retry
                                 if end_thinking(msg) → deliver msg, exit
                                 execute_tool(name, args) → result
                                 → optional file delivery
                               if all providers fail → live 30s countdown

BackgroundAgentManager (singleton)
├── Watcher asyncio.Tasks  (pure Python, zero LLM tokens)
└── WakeProcessor task     (1 LLM call per fired signal)

ProviderPool (singleton)
├── Keyed:    Gemini · Groq · OpenRouter · NVIDIA NIM · Vercel AI
└── Keyless:  Ollama · G4F

Storage
├── SQLite / Postgres  users · api_keys [AES-256-GCM] · chat_history
│                      rika_memory · background_agents · wake_events · command_audit
└── Qdrant             collection: collective_unconscious
```

---

## Project layout

```
rika-agent/
├── src/
│   ├── agents/
│   │   ├── background/         zero-token watchers + WakeProcessor
│   │   ├── agent_factory.py    ConcreteAgent — function-calling ReAct loop
│   │   ├── agent_bus.py        parallel + dependency-ordered runner
│   │   └── agent_models.py     AgentSpec, WakeSignal, BackgroundAgentConfig
│   ├── bot/app.py              all handlers, auth gate, orchestration loop
│   ├── core/
│   │   ├── orchestrator.py     Orchestrator class (v2.2.0 target)
│   │   ├── complexity.py       3-tier message classifier
│   │   ├── context.py          per-turn context assembler
│   │   ├── event_bus.py        SessionEvent pub/sub (v2.2.0 target)
│   │   └── models.py           shared data structures
│   ├── db/
│   │   ├── connection.py       dual SQLite/Postgres backend
│   │   ├── migrate.py          migration runner (idempotent)
│   │   ├── pg_compat.py        asyncpg shim for SQLite-style queries
│   │   ├── pg_migrate.py       consolidated Postgres schema
│   │   ├── vercel_kv.py        Upstash Redis client
│   │   ├── chat_store.py
│   │   ├── key_store.py
│   │   ├── background_store.py
│   │   └── vector_store.py     Qdrant + fastembed
│   ├── providers/
│   │   ├── base_provider.py    StructuredResponse, ToolCall, abstract base
│   │   ├── gemini_provider.py  function calling + vision + Part wrappers
│   │   ├── groq_provider.py    function calling + enum stripping
│   │   ├── openrouter_provider.py
│   │   ├── nvidia_provider.py  NVIDIA NIM (OpenAI-compatible)
│   │   ├── vercel_provider.py  Vercel AI Gateway (OpenAI-compatible)
│   │   ├── ollama_provider.py  local LLM
│   │   ├── g4f_provider.py     free endpoints
│   │   └── provider_pool.py    singleton, key rotation, cap enforcement, failover
│   ├── tools/
│   │   ├── schemas.py          JSON Schema for all tools (including end_thinking)
│   │   ├── sandbox.py          3-level code isolation
│   │   ├── command_security.py 22-rule shell firewall
│   │   ├── shell_tool.py       async shell + secondary injection block + audit log
│   │   ├── web_search_tool.py  DuckDuckGo scraper with GET fallback
│   │   └── curl_tool.py        URL fetch (raises on 5xx only)
│   └── utils/
│       ├── parse_keys.py       /addkey parser (all 7 providers)
│       └── retry.py            exponential backoff utility
├── scripts/
│   ├── bot_setup.sh            interactive setup wizard
│   └── start.sh                pre-flight + launch
├── config.json.template
├── soul.md.template
├── .env.template
├── docker-compose.yml
├── ROADMAP.md
└── README.md
```

---


## Security

<details>
<summary>Owner-only auth gate</summary>

Set `OWNER_USER_ID` in `.env` (your Telegram user ID — send any message to [@userinfobot](https://t.me/userinfobot) to find it). The bot will silently ignore all messages from anyone else — no reply, no error, no indication the bot exists.

```env
OWNER_USER_ID=123456789
```

If `OWNER_USER_ID` is unset, the bot responds to all Telegram users. This is fine for local development but should not be used if the bot is reachable from the internet.

</details>

<details>
<summary>API key encryption</summary>

AES-256-GCM. Generate your encryption key:
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```
The key never touches the database. If `.env` is lost, stored keys are unrecoverable.

</details>

<details>
<summary>Shell command firewall</summary>

22 rules evaluated before every `run_shell_command` call. Zero AI tokens, zero latency. Plus a secondary structural check that blocks `$()` subshell substitution, backtick substitution, and `;rm/curl/wget/bash` chains unconditionally regardless of security level.

Three security levels in `config.json`:
- `"standard"` — CRITICAL + HIGH blocked, MEDIUM needs `CONFIRM:` prefix
- `"strict"` — also blocks `curl | bash`, `sudo su`, service-disabling
- `"permissive"` — CRITICAL only (trusted personal use only)

</details>

<details>
<summary>File delivery</summary>

`send_file` and `read_file` resolve symlinks and reject any path outside `~/.rika/shared` using `Path.resolve()` + `is_relative_to()`. The old `lstrip("/").replace("..")` approach is gone — the new check is not bypassable with `....//` tricks.

</details>

---

## Contributing

Issues and PRs welcome. For security issues, open a private issue rather than disclosing publicly.

---

<div align="center">

Built in Zagora, Morocco, by [piratheon](https://github.com/piratheon) — inspired by the works of [Lain](https://lainchan.org/), [Rei](https://www.reddit.com/r/ReiAyanami/), and countless cyberpunk dreamers xD!

</div>
