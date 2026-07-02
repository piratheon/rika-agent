# skill: web_search
description: Search the web for current information, news, and real-time data.
tools: web_search, curl, wikipedia_search

## When to use
- Current events, recent news, live prices, weather
- Anything that might have changed since training cutoff
- Verifying facts from multiple sources

## Usage pattern
1. Call `web_search(query="...")` with a concise query
2. For deeper content on a result, call `curl(url="...")` on the link
3. For established facts or history, prefer `wikipedia_search` over web_search

## Notes
- max_results default is 5; increase to 10 for research tasks
- DuckDuckGo backend — no login required
