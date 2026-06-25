import httpx
from src.utils.logger import logger

async def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia and return a detailed summary.
    Uses the Official MediaWiki Action API.
    """
    search_url = "https://en.wikipedia.org/w/api.php"
    
    # 1. Search for the most relevant page
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 1
    }
    
    headers = {"User-Agent": "AgentBot/1.0 (https://github.com/piratheon/Rika-Agent)"}
    
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            # Step 1: Find best match
            r = await client.get(search_url, params=search_params, headers=headers)
            r.raise_for_status()
            search_data = r.json()
            
            search_results = search_data.get("query", {}).get("search", [])
            if not search_results:
                return f"Wikipedia: No results found for '{query}'."
            
            page_title = search_results[0]["title"]
            
            # Step 2: Get detailed summary for that page
            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title.replace(' ', '_')}"
            r_summary = await client.get(summary_url, headers=headers)
            
            if r_summary.status_code == 200:
                data = r_summary.json()
                extract = data.get("extract", "No summary available.")
                return f"Wikipedia ({page_title}): {extract}"
            else:
                # Fallback to the search snippet
                snippet = search_results[0].get("snippet", "").replace('<span class="searchmatch">', '').replace('</span>', '')
                return f"Wikipedia ({page_title} - snippet): {snippet}..."
                
    except Exception as e:
        logger.error("wikipedia_real_api_failed", error=str(e))
        return f"Wikipedia Error: {str(e)}"
