import wikipedia
import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class WikipediaSearchInput(BaseModel):
	query: str = Field(description="Topic to look up on Wikipedia")
	sentences: int = Field(default=3, ge=1, le=10, description="Number of sentences")


class MultiplyInput(BaseModel):
	a: float = Field(description="First number")
	b: float = Field(description="Second number")


class DuckDuckGoSearchInput(BaseModel):
	query: str = Field(description="Web search query")
	max_results: int = Field(default=5, ge=1, le=10, description="Maximum result count")


class FetchWebPageInput(BaseModel):
    url: str = Field(description="Full URL of the web page to fetch")
    max_chars: int = Field(default=4000, ge=100, le=20000, description="Maximum characters of text to return")


@tool(args_schema=WikipediaSearchInput)
def wikipedia_search(query: str, sentences: int = 3) -> str:
    """Search Wikipedia and return a short summary for a topic."""
    print(f"\n[tool] wikipedia_search called with query='{query}'")
    try:
        result = wikipedia.summary(query, sentences=sentences, auto_suggest=True)
        first_line = result.splitlines()[0] if result else "(no result)"
        print(f"[tool] wikipedia_search result preview: {first_line}")
        return result
    except Exception as error:
        print(f"[tool] wikipedia_search error: {error}")
        return f"Wikipedia search failed: {error}"


@tool(args_schema=MultiplyInput)
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    print(f"\n[tool] multiply called with a={a}, b={b}")
    result = a * b
    print(f"[tool] multiply result: {result}")
    return result


@tool(args_schema=DuckDuckGoSearchInput)
def duckduckgo_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search the web using DuckDuckGo and return structured results."""
    print(f"\n[tool] duckduckgo_search called with query='{query}'")
    try:
        with DDGS() as ddgs:
            raw_results = ddgs.text(query, max_results=max_results)
            results: list[dict[str, str]] = []
            for item in raw_results:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("href", ""),
                        "snippet": item.get("body", ""),
                    }
                )
            first = results[0] if results else {}
            print(f"[tool] duckduckgo_search result preview: {first.get('title', '(no result)')} — {first.get('url', '')}")
            return results
    except Exception as error:
        print(f"[tool] duckduckgo_search error: {error}")
        return [{"title": "Search error", "url": "", "snippet": str(error)}]


@tool(args_schema=FetchWebPageInput)
def fetch_webpage(url: str, max_chars: int = 4000) -> str:
    """Fetch a web page by URL and return its readable text content."""
    print(f"\n[tool] fetch_webpage called with url='{url}'")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; research-agent/1.0)"}
        response = httpx.get(url, headers=headers, timeout=10, follow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        trimmed = text[:max_chars]
        first_line = trimmed.splitlines()[0] if trimmed else "(empty)"
        print(f"[tool] fetch_webpage result preview: {first_line}")
        return trimmed
    except Exception as error:
        print(f"[tool] fetch_webpage error: {error}")
        return f"Failed to fetch page: {error}"


TOOLS = [wikipedia_search, multiply, duckduckgo_search, fetch_webpage]
