"""Load text content from PDF files."""
from pathlib import Path
from typing import List

from pypdf import PdfReader

from rag.schema import Document


class PDFLoader:
    """Loads a PDF file into Document objects, one per page by default so
    citations can point back to a specific page."""

    def __init__(self, split_by_page: bool = True):
        self.split_by_page = split_by_page

    def load(self, file_path: str) -> List[Document]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        reader = PdfReader(str(path))
        documents: List[Document] = []

        if self.split_by_page:
            for i, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()
                if not text:
                    continue
                documents.append(
                    Document(
                        content=text,
                        metadata={"source": path.name, "page": i + 1, "type": "pdf"},
                    )
                )
        else:
            full_text = "\n".join((p.extract_text() or "") for p in reader.pages)
            documents.append(
                Document(content=full_text.strip(), metadata={"source": path.name, "type": "pdf"})
            )

        return documents
