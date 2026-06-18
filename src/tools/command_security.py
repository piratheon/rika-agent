"""Command security checker — pure Python, zero AI tokens.

Multi-layer analysis:
  1. Exact-match blocklist for known one-liner disasters
  2. Regex pattern analysis for structural danger
  3. Critical path protection (system directories)
  4. Pipe-to-shell exfiltration detection
  5. Privilege escalation detection
  6. Resource exhaustion patterns (fork bombs, disk wipes)

Severity levels:
  CRITICAL  — blocked unconditionally (irreversible system damage)
  HIGH      — blocked by default (serious risk, rarely legitimate)
  MEDIUM    — warned + confirmation required (risky but sometimes valid)
  LOW       — logged only (suspicious but allow through)
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SecurityCheckResult:
    allowed: bool
    severity: str              # CRITICAL | HIGH | MEDIUM | LOW | OK
    reason: str = ""
    suggestion: str = ""
    matched_rule: str = ""
    requires_confirmation: bool = False   # True for MEDIUM — ask user first


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

@dataclass
class SecurityRule:
    name: str
    pattern: re.Pattern
    severity: str
    reason: str
    suggestion: str = ""


def _r(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE | re.DOTALL)


# Critical system paths that should never be targets of destructive operations
_CRITICAL_PATHS = (
    r"(?:^|[\s/])(?:/(?:etc|boot|usr|bin|sbin|lib|lib64|sys|proc|dev|run)(?:/|$)|/$)"
)

_RULES: List[SecurityRule] = [

    # -----------------------------------------------------------------------
    # CRITICAL — block unconditionally
    # -----------------------------------------------------------------------

    SecurityRule(
        name="rm_root",
        pattern=_r(r"\brm\b.*?(?:-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r|--recursive.*--force|--force.*--recursive).*?(?:/\s*$|/\*|/\s+\*|\"/?\"|\s+/\b)"),
        severity="CRITICAL",
        reason="Recursive force-delete targeting root or wildcard — would wipe the filesystem.",
        suggestion="Specify the exact path you want to delete, e.g.: rm -rf ~/.rika/shared/tmp/",
    ),
    SecurityRule(
        name="rm_no_preserve_root",
        pattern=_r(r"\brm\b.*--no-preserve-root"),
        severity="CRITICAL",
        reason="--no-preserve-root disables the safety guard on rm, allowing deletion of /.",
        suggestion="Never use --no-preserve-root unless you intend to wipe the system.",
    ),
    SecurityRule(
        name="fork_bomb",
        pattern=_r(r":\(\)\s*\{.*:\|.*:.*&.*\}|:\(\)\{:\|:\&\};:"),
        severity="CRITICAL",
        reason="Fork bomb detected — would exhaust system PIDs and freeze the machine.",
        suggestion="There is no legitimate use for a fork bomb.",
    ),
    SecurityRule(
        name="disk_wipe_dd",
        pattern=_r(r"\bdd\b.*of\s*=\s*/dev/(?:sd[a-z]|nvme\d|hd[a-z]|vd[a-z]|xvd[a-z])\b(?!\d*p\d)"),
        severity="CRITICAL",
        reason="dd writing directly to a block device — would destroy all data on that disk.",
        suggestion="dd to a file or image, not a raw device: dd if=... of=~/backup.img",
    ),
    SecurityRule(
        name="mkfs_device",
        pattern=_r(r"\bmkfs(?:\.\w+)?\s+/dev/(?:sd[a-z]|nvme\d|hd[a-z]|vd[a-z])"),
        severity="CRITICAL",
        reason="mkfs formats a raw disk device — all data would be lost immediately.",
        suggestion="Only format partitions (e.g. /dev/sdb1) and only when you intend to.",
    ),
    SecurityRule(
        name="shred_device",
        pattern=_r(r"\bshred\b.*?/dev/(?:sd[a-z]|nvme\d|hd[a-z])"),
        severity="CRITICAL",
        reason="shred on a block device securely wipes the entire disk.",
        suggestion="shred individual files, not whole block devices.",
    ),
    SecurityRule(
        name="kernel_panic",
        pattern=_r(r"echo\s+['\"]?c['\"]?\s*>\s*/proc/sysrq-trigger"),
        severity="CRITICAL",
        reason="Writing 'c' to sysrq-trigger triggers an immediate kernel panic.",
        suggestion="Never write to /proc/sysrq-trigger.",
    ),
    SecurityRule(
        name="overwrite_passwd",
        pattern=_r(r">\s*/etc/(?:passwd|shadow|sudoers|group|gshadow|hosts)\b"),
        severity="CRITICAL",
        reason="Overwriting critical authentication files would lock out all users.",
        suggestion="Edit auth files with vipw/visudo — never redirect into them.",
    ),
    SecurityRule(
        name="chmod_world_root",
        pattern=_r(r"\bchmod\b.*?(?:-R|--recursive).*?(?:777|a\+w).*?(?:/\s*$|/etc|/usr|/bin|/sbin)"),
        severity="CRITICAL",
        reason="Recursive chmod 777 on system paths creates severe security vulnerabilities.",
        suggestion="chmod only specific files with the minimum permissions needed.",
    ),
    SecurityRule(
        name="dev_null_redirect",
        pattern=_r(r"/dev/null\s*>\s*/(?:etc|bin|sbin|usr|boot|lib)\b"),
        severity="CRITICAL",
        reason="Redirecting /dev/null into system binaries or config files destroys them.",
        suggestion="",
    ),

    # -----------------------------------------------------------------------
    # HIGH — blocked by default
    # -----------------------------------------------------------------------

    SecurityRule(
        name="kill_init",
        pattern=_r(r"\bkill\b.*?(?:-9|-KILL|-SIGKILL)\s+1\b|\bkillall\s+init\b|\bkillall\s+systemd\b"),
        severity="HIGH",
        reason="Killing PID 1 (init/systemd) will immediately crash the system.",
        suggestion="Use systemctl or service commands to manage services individually.",
    ),
    SecurityRule(
        name="flush_iptables",
        pattern=_r(r"\biptables\s+-F\b|\biptables\s+--flush\b|\bnft\s+flush\s+ruleset\b"),
        severity="HIGH",
        reason="Flushing all firewall rules drops all network security policies immediately.",
        suggestion="Remove specific rules instead: iptables -D CHAIN rule-number",
    ),
    SecurityRule(
        name="overwrite_grub",
        pattern=_r(r"\bdd\b.*of\s*=\s*/dev/(?:sd[a-z]|nvme\d)\s*\b.*(?:bs=512|count=1)|grub.*--force.*(?:/dev/sd|/dev/nvme)"),
        severity="HIGH",
        reason="Writing to the MBR/bootloader sector would make the system unbootable.",
        suggestion="Use grub-install only if you know what you are doing and have a backup.",
    ),
    SecurityRule(
        name="rm_etc_critical",
        pattern=_r(r"\brm\b.*?(?:-[rf]+).*?/etc/(?:passwd|shadow|sudoers|fstab|hosts|ssh)"),
        severity="HIGH",
        reason="Deleting critical system configuration files would break authentication/networking.",
        suggestion="Back up files before removing: cp /etc/file /etc/file.bak",
    ),
    SecurityRule(
        name="chmod_suid_root",
        pattern=_r(r"\bchmod\b.*?[u+]?s.*?/(?:bin|usr/bin|sbin)/"),
        severity="HIGH",
        reason="Setting SUID on system binaries is a common privilege escalation vector.",
        suggestion="Do not set SUID bits on binaries unless absolutely necessary.",
    ),

    # -----------------------------------------------------------------------
    # MEDIUM — warn and require user confirmation
    # -----------------------------------------------------------------------

    SecurityRule(
        name="pipe_to_shell",
        pattern=_r(r"(?:curl|wget|fetch)\b[^|]*\|[^|]*(?:bash|sh|zsh|fish|python|ruby|perl)\b"),
        severity="MEDIUM",
        reason="Piping remote content directly into a shell executes untrusted code.",
        suggestion="Download first and inspect: curl -o script.sh URL && cat script.sh && bash script.sh",
    ),
    SecurityRule(
        name="sudo_su",
        pattern=_r(r"\bsudo\s+su\b|\bsudo\s+-i\b|\bsudo\s+bash\b|\bsudo\s+sh\b|\bsu\s+-\s*root\b"),
        severity="MEDIUM",
        reason="Switching to a root shell removes all access controls for the session.",
        suggestion="Use sudo <specific-command> instead of gaining a full root shell.",
    ),
    SecurityRule(
        name="rm_home_recursive",
        pattern=_r(r"\brm\b.*?-[a-zA-Z]*r[a-zA-Z]*.*?~/(?!\s*$|\.\w)|\brm\b.*?-[a-zA-Z]*r[a-zA-Z]*.*?/home/\w+\s*$"),
        severity="MEDIUM",
        reason="Recursive deletion of a home directory would remove all user files.",
        suggestion="Double-check the target path. Consider moving to trash instead.",
    ),
    SecurityRule(
        name="systemctl_disable_critical",
        pattern=_r(r"\bsystemctl\b.*?(?:disable|mask|stop)\s+(?:ssh|sshd|networking|network|NetworkManager|systemd-networkd|cron|crond|firewalld|ufw)\b"),
        severity="MEDIUM",
        reason="Disabling critical system services may make the system unreachable.",
        suggestion="Test service changes in a recovery-accessible environment first.",
    ),
    SecurityRule(
        name="write_crontab_root",
        pattern=_r(r"crontab\s+-r\b"),
        severity="MEDIUM",
        reason="crontab -r removes all cron jobs for the current user with no confirmation.",
        suggestion="List jobs first with crontab -l, then edit with crontab -e to remove specific entries.",
    ),
]

# Pre-compiled for performance
_COMPILED_RULES = _RULES


# ---------------------------------------------------------------------------
# Workspace path helper
# ---------------------------------------------------------------------------

def _is_within_workspace(path: str, workspace: str) -> bool:
    """Return True if path is inside the agent workspace (safe to modify)."""
    import os
    try:
        real_path = os.path.realpath(os.path.expanduser(path))
        real_ws = os.path.realpath(os.path.expanduser(workspace))
        return real_path.startswith(real_ws)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main check function
# ---------------------------------------------------------------------------

def check_command(
    command: str,
    workspace_path: str = "~/.rika/shared",
    security_level: str = "standard",
) -> SecurityCheckResult:
    """Check a shell command against the security ruleset.

    Args:
        command:        The raw shell command string.
        workspace_path: Agent's working directory (operations here are safer).
        security_level: "standard" (CRITICAL+HIGH blocked, MEDIUM warned)
                        "strict"   (also blocks pipe-to-shell and sudo-su)
                        "permissive" (CRITICAL only)

    Returns:
        SecurityCheckResult
    """
    if not command or not command.strip():
        return SecurityCheckResult(allowed=True, severity="OK")

    cmd = command.strip()

    for rule in _COMPILED_RULES:
        if not rule.pattern.search(cmd):
            continue

        sev = rule.severity

        # Determine block/warn based on severity + security_level
        if sev == "CRITICAL":
            return SecurityCheckResult(
                allowed=False,
                severity=sev,
                reason=rule.reason,
                suggestion=rule.suggestion,
                matched_rule=rule.name,
            )

        if sev == "HIGH":
            blocked = security_level in ("standard", "strict")
            return SecurityCheckResult(
                allowed=not blocked,
                severity=sev,
                reason=rule.reason,
                suggestion=rule.suggestion,
                matched_rule=rule.name,
                requires_confirmation=not blocked,
            )

        if sev == "MEDIUM":
            if security_level == "strict":
                return SecurityCheckResult(
                    allowed=False,
                    severity=sev,
                    reason=rule.reason,
                    suggestion=rule.suggestion,
                    matched_rule=rule.name,
                )
            # standard/permissive: warn and require confirmation
            return SecurityCheckResult(
                allowed=False,
                severity=sev,
                reason=rule.reason,
                suggestion=rule.suggestion,
                matched_rule=rule.name,
                requires_confirmation=True,
            )

        if sev == "LOW":
            # Log but allow through
            return SecurityCheckResult(
                allowed=True,
                severity=sev,
                reason=rule.reason,
                suggestion=rule.suggestion,
                matched_rule=rule.name,
            )

    return SecurityCheckResult(allowed=True, severity="OK")


def format_block_message(result: SecurityCheckResult) -> str:
    """Format a human-readable block message for Telegram."""
    icon = {"CRITICAL": "BLOCKED", "HIGH": "BLOCKED", "MEDIUM": "WARNING"}.get(result.severity, "NOTE")
    lines = [f"[{icon}] Command refused — {result.severity}"]
    if result.reason:
        lines.append(f"\nReason: {result.reason}")
    if result.suggestion:
        lines.append(f"\nSuggestion: {result.suggestion}")
    if result.requires_confirmation:
        lines.append("\nSend the command again with 'CONFIRM: ' prefix to execute anyway.")
    return "\n".join(lines)
