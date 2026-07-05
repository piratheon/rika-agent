# skill: web_scraping
description: Extract structured data from web pages using httpx + BeautifulSoup or playwright for JS-heavy sites.
tools: run_python, run_shell_command

## When to use
- Extracting tables, lists, or article text from static HTML pages
- Scraping data that has no API
- Handling JS-rendered pages with Playwright when static scraping fails

## Usage pattern
```python
# Static page scraping (BeautifulSoup)
run_python(code="""
import httpx
from bs4 import BeautifulSoup

r = httpx.get("https://example.com/data", timeout=15,
              headers={"User-Agent": "Mozilla/5.0"})
r.raise_for_status()
soup = BeautifulSoup(r.text, "html.parser")

# Extract table rows
for row in soup.select("table tbody tr"):
    cols = [td.get_text(strip=True) for td in row.find_all("td")]
    print(cols)
""")

# JS-rendered page (Playwright)
run_python(code="""
from playwright.sync_api import sync_playwright

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("https://example.com/spa", wait_until="networkidle")
    items = page.query_selector_all(".item")
    for item in items:
        print(item.inner_text())
    browser.close()
""")
```

## Notes
- Install: `pip install httpx beautifulsoup4 lxml playwright && playwright install chromium`
- Prefer BeautifulSoup for static sites — it is faster and lighter
- Use `lxml` parser for speed: `BeautifulSoup(html, "lxml")`
- Respect robots.txt; add a 1s delay between requests on the same domain
