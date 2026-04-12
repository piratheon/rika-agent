import httpx
import re
from src.utils.logger import logger

async def curl_fetch(url: str) -> str:
    """
    Powerful web fetcher (free-form curl equivalent).
    Cleans up HTML and returns readable text content.
    Supports 'insecure' mode via '--insecure' or '-k' in the URL string (heuristic).
    """
    url = url.strip()
    verify_ssl = True
    
    # Heuristic: if agent adds -k or --insecure to the "url"
    if " -k" in url or " --insecure" in url:
        verify_ssl = False
        url = url.replace(" -k", "").replace(" --insecure", "").strip()

    if not url.startswith("http"):
        url = "https://" + url
        
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    
    try:
        # Use verify=False if insecure mode is requested
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, verify=verify_ssl) as client:
            r = await client.get(url, headers=headers)
            # Only raise on server errors (5xx). 4xx may still have content.
            if r.status_code >= 500:
                r.raise_for_status()
            
            text = r.text
            
            # 1. Remove non-content tags
            text = re.sub(r'<(script|style|header|footer|nav|noscript).*?>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)
            
            # 2. Extract title
            title_match = re.search(r'<title>(.*?)</title>', text, re.IGNORECASE)
            title = title_match.group(1) if title_match else "No Title"
            
            # 3. Strip tags
            text = re.sub(r'<.*?>', ' ', text)
            
            # 4. Cleanup whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            preview = text[:2500]
            return f"FETCHED CONTENT FROM: {url}\nTITLE: {title}\nVERIFY_SSL: {verify_ssl}\nCONTENT: {preview}..."
            
    except httpx.ConnectError as e:
        return f"CURL ERROR: Connection failed. Check if URL is valid: {url}"
    except httpx.HTTPStatusError as e:
        return f"CURL ERROR: Server returned {e.response.status_code} for {url}"
    except Exception as e:
        err_msg = str(e)
        if "CERTIFICATE_VERIFY_FAILED" in err_msg:
            return f"CURL ERROR: SSL Certificate verification failed for {url}. Hint: Use 'TOOL: curl | QUERY: {url} --insecure' to skip verification."
        logger.error("curl_tool_failed", url=url, error=err_msg)
        return f"CURL ERROR: {err_msg}"
