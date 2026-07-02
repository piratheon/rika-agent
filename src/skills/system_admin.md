# skill: system_admin
description: System administration — process management, service status, resource monitoring, log inspection.
tools: run_shell_command

## When to use
- Checking CPU, memory, disk usage
- Inspecting running processes or services
- Viewing system logs
- Managing cron jobs

## Usage pattern
```bash
# Resource snapshot
run_shell_command("free -h && df -h && uptime")

# Process list
run_shell_command("ps aux --sort=-%mem | head -20")

# Service status (systemd)
run_shell_command("systemctl status servicename --no-pager -l")

# Last 50 lines of a log
run_shell_command("tail -50 /var/log/syslog")

# Active cron jobs for current user
run_shell_command("crontab -l 2>/dev/null || echo 'no crontab'")
```

## Notes
- Destructive commands (kill -9, rm -rf system paths) require CONFIRM: prefix
- For Docker: `docker ps`, `docker logs container`, `docker stats`
