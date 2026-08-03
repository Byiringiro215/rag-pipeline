"""Load text content from HTML files or raw HTML strings."""
from pathlib import Path
from typing import List, Optional

from bs4 import BeautifulSoup

from rag.schema import Document


class HTMLLoader:
    """Loads and cleans HTML content into plain-text Documents, stripping
    non-content elements like scripts, nav bars, and footers."""

    def __init__(self, strip_selectors: Optional[List[str]] = None):
        self.strip_selectors = strip_selectors or ["script", "style", "nav", "footer", "header", "noscript"]

    def _clean(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for selector in self.strip_selectors:
            for tag in soup.select(selector):
                tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    def load(self, file_path: str) -> List[Document]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"HTML file not found: {file_path}")
        html = path.read_text(encoding="utf-8", errors="ignore")
        text = self._clean(html)
        return [Document(content=text, metadata={"source": path.name, "type": "html"})]

    def load_string(self, html: str, source_name: str = "raw_html") -> List[Document]:
        """Useful for ingesting scraped/fetched HTML that isn't on disk."""
        text = self._clean(html)
        return [Document(content=text, metadata={"source": source_name, "type": "html"})]
