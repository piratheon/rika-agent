# skill: ssh_remote
description: Run commands on remote hosts over SSH, transfer files with scp/rsync.
tools: run_shell_command

## When to use
- Executing commands on a remote server
- Copying files to or from a remote host
- Setting up SSH key authentication
- Port-forwarding for local access to remote services

## Usage pattern
```bash
# Run a single command remotely (key-based auth assumed)
run_shell_command("ssh -o StrictHostKeyChecking=no user@host 'uptime && df -h'")

# Copy local file to remote
run_shell_command("scp -o StrictHostKeyChecking=no ./script.sh user@host:/tmp/")

# Copy remote file locally
run_shell_command("scp user@host:/var/log/app.log ./app.log")

# Rsync (incremental, preserves permissions)
run_shell_command("rsync -avz --progress ./dist/ user@host:/var/www/app/")

# Local port-forward (background)
run_shell_command("ssh -fNL 5432:localhost:5432 user@host")

# Run a script file remotely
run_shell_command("ssh user@host 'bash -s' < ./setup.sh")
```

## Notes
- SSH keys must be configured; passwords are not supported in non-interactive mode
- Use `-i /path/to/key` to specify an identity file
- Add `ConnectTimeout=10` to avoid long hangs: `-o ConnectTimeout=10`
- For jump hosts: `-J jumpuser@jumphost targetuser@targethost`
- Rsync dry-run: add `--dry-run` to preview changes before applying
