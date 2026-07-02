# skill: git
description: Git operations within the workspace — clone, pull, diff, commit, branch, log.
tools: run_shell_command

## When to use
- Cloning a repository into the workspace for analysis or modification
- Creating branches, committing changes, viewing history
- Diffing files before committing

## Usage pattern
```bash
# Clone a repository
run_shell_command("git clone https://github.com/user/repo.git")

# Stage and commit changes
run_shell_command("git -C repo/ add -A && git -C repo/ commit -m 'description'")

# View recent log
run_shell_command("git -C repo/ log --oneline -20")

# Diff before commit
run_shell_command("git -C repo/ diff --stat HEAD")
```

## Security constraints
- push and remote add are not permitted without explicit user confirmation
- All paths must remain within ~/.rika/shared
