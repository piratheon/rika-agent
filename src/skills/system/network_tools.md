# skill: network_tools
description: Diagnose network connectivity, inspect ports, resolve DNS, and measure latency.
tools: run_shell_command

## When to use
- Checking if a host or port is reachable
- Resolving DNS or tracing routing hops
- Inspecting active connections or listening ports
- Measuring download speed or HTTP response times

## Usage pattern
```bash
# Check connectivity
run_shell_command("ping -c 4 8.8.8.8")
run_shell_command("curl -o /dev/null -s -w '%{http_code} %{time_total}s\n' https://example.com")

# Port check (nc is faster than nmap for single ports)
run_shell_command("nc -zv -w3 example.com 443 2>&1")

# DNS lookup
run_shell_command("dig +short example.com A")
run_shell_command("dig +short example.com MX")

# Trace route
run_shell_command("traceroute -n -w2 example.com")

# Active listening ports
run_shell_command("ss -tlnp")

# Active connections to a specific port
run_shell_command("ss -tnp sport = :443")

# HTTP header inspection
run_shell_command("curl -sI https://example.com | head -20")
```

## Notes
- `ss` replaces deprecated `netstat` on modern Linux
- For TLS cert inspection: `echo | openssl s_client -connect host:443 2>/dev/null | openssl x509 -noout -dates`
- nmap scan (if installed): `nmap -sV -p 22,80,443 host` — only on hosts you own or have permission to scan
