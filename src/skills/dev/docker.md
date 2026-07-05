# skill: docker
description: Build, run, inspect, and manage Docker containers and images.
tools: run_shell_command

## When to use
- Running services or code in isolated containers
- Building images from a Dockerfile
- Inspecting container logs, stats, or network
- Managing volumes and compose stacks

## Usage pattern
```bash
# Build image
run_shell_command("docker build -t myapp:latest .")

# Run container (detached, port-mapped)
run_shell_command("docker run -d -p 8080:8080 --name myapp myapp:latest")

# Follow logs
run_shell_command("docker logs -f myapp --tail 50")

# Execute shell inside running container
run_shell_command("docker exec -it myapp /bin/sh")

# Compose stack
run_shell_command("docker compose up -d")
run_shell_command("docker compose logs --tail 30")
run_shell_command("docker compose down")

# Cleanup unused resources
run_shell_command("docker system prune -f")
```

## Notes
- Never pass secrets as -e flags in shell history — use --env-file instead
- Use `docker inspect <name>` for full container metadata
- `docker stats --no-stream` for a one-shot resource snapshot
