# skill: json_yaml
description: Parse, transform, query, and format JSON and YAML data using jq, yq, or Python.
tools: run_shell_command, run_python

## When to use
- Extracting fields from complex JSON/YAML responses
- Transforming data structures between formats
- Querying nested data without writing full Python parsers

## Usage pattern
```bash
# jq — extract field from JSON file
run_shell_command("jq '.items[].name' data.json")

# jq — filter array where condition is true
run_shell_command("jq '[.users[] | select(.active == true)]' users.json")

# jq — reformat / compact
run_shell_command("jq -c '.' large.json | head -5")

# yq — read a YAML field
run_shell_command("yq '.services.web.image' docker-compose.yml")

# yq — update a field in-place
run_shell_command("yq -i '.version = "2.1"' config.yml")
```

```python
# Python fallback when jq/yq unavailable
run_python(code="""
import json, pathlib

data = json.loads(pathlib.Path("data.json").read_text())
results = [item["name"] for item in data["items"] if item.get("active")]
print(results)
""")
```

## Notes
- Install jq: `apt-get install jq` or `brew install jq`
- Install yq (Python): `pip install yq`  |  (Go binary): `brew install yq`
- Pipe jq output through `| head` for large files to avoid flooding context
