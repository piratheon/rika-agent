# skill: api_client
description: Make authenticated REST API calls; handle pagination, rate limits, and JSON responses.
tools: run_python, run_shell_command

## When to use
- Calling third-party REST APIs with API keys or OAuth tokens
- Paginating through large API responses
- Transforming JSON responses into usable data

## Usage pattern
```python
# Basic authenticated GET
run_python(code="""
import httpx, json

headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
with httpx.Client(timeout=30) as client:
    r = client.get("https://api.example.com/v1/resource", headers=headers, params={"limit": 50})
    r.raise_for_status()
    data = r.json()
print(json.dumps(data, indent=2)[:2000])
""")

# Paginated GET (offset-based)
run_python(code="""
import httpx

results = []
page = 1
with httpx.Client(timeout=30) as client:
    while True:
        r = client.get("https://api.example.com/items",
                        headers={"Authorization": "Bearer TOKEN"},
                        params={"page": page, "per_page": 100})
        r.raise_for_status()
        batch = r.json().get("items", [])
        if not batch:
            break
        results.extend(batch)
        page += 1
print(f"Total fetched: {len(results)}")
""")

# POST with JSON body
run_python(code="""
import httpx
r = httpx.post("https://api.example.com/v1/create",
               headers={"Authorization": "Bearer TOKEN"},
               json={"name": "value", "active": True},
               timeout=30)
r.raise_for_status()
print(r.json())
""")
```

## Notes
- Install: `pip install httpx`
- Always call `r.raise_for_status()` before accessing `.json()`
- For OAuth2: exchange client_credentials at /oauth/token, cache the token
- Rate limits: check `Retry-After` header on 429 responses
