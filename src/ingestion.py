"""Load source documents for the RAG pipeline."""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_all_pdfs(folder_path: str | Path = "data/pdfs"):
    """Load every PDF in a folder and return LangChain Document pages."""
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(
            f"PDF folder not found: {folder}. Create it and add PDF files."
        )
    if not folder.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {folder}")

    pdf_files = sorted(folder.glob("*.pdf")) + sorted(folder.glob("*.PDF"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {folder}")

    all_documents = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        all_documents.extend(loader.load())

    print(f"Loaded {len(pdf_files)} PDF(s) and {len(all_documents)} page(s).")
    return all_documents


def load_text_file(file_path: str | Path, encoding: str = "utf-8"):
    """Load a plain-text file and return LangChain Document objects."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Text file not found: {path}")
    return TextLoader(str(path), encoding=encoding).load()


def split_docs(documents, chunk_size: int = 500, chunk_overlap: int = 50):
    """Split documents into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size.")

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)
