from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document

from app.monitoring import get_logger
from app.embeddings import create_embeddings
from app.vector_store import get_vector_store

logger = get_logger()

DATA_DIRECTORY = "./data"

def load_files(path: str = DATA_DIRECTORY):
    """Load all pdf documents from a directory."""

    if not Path(path).exists():
        logger.error(f"Directory not found: {path}")
        raise FileNotFoundError(f"Directory not found: {path}")
    
    # load all .pdf files in a directory
    try:
        loader = DirectoryLoader(
            path,
            glob="*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents = loader.load()

        for doc in documents:
            doc.metadata["source"] = Path(doc.metadata["source"]).name

        logger.info(f"Loaded {len(documents)} documents from {path}/")
        return documents

    except Exception as e:
        logger.error(f"Error loading {path}: {e}")
        return []
    
def chunk_documents(
        documents: list[Document], chunk_size: int = 400, chunk_overlap: int= 100
    ) -> list[Document]:
    """Split documents into chunks for better retrieval.

    Variable chunking strategy:
    - Default: 400 characters with 100 overlap

    Args:
        documents: list of documents to chunk
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks

    Returns:
        list[Document]: Chunked documents with preserved metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunked = text_splitter.split_documents(documents)

    logger.info(f"Split {len(documents)} documents into {len(chunked)} chunks")
    logger.info(f"Avg chunk size: {sum(len(c.page_content) for c in chunked) // len(chunked)} chars")

    return chunked

def build_and_load_knowledge_base() -> dict:
    """Build complete knowledge base and load into Supabase.

    Returns:
        dict: Statistics about the loaded knowledge base
    """
    documents = load_files()

    if len(documents) == 0:
        logger.error("No documents loaded! Aborting.")
        raise ValueError("No documents loaded.")
    
    chunked_docs = chunk_documents(documents)

    try: 
        embeddings = create_embeddings()
        logger.info("OpenAI embeddings initialized successfully")
    except Exception as e:
        logger.error(f"Embeddings initialization failed: {e}")

    try:
        vectorstore = get_vector_store(embeddings)
        logger.info("Qdrant vector store initialized")
    except Exception as e:
        logger.error(f"Vector store initialization failed: {e}")

    try:
        batch_size = 100
        total_loaded = 0

        for i in range(0, len(chunked_docs), batch_size):
            batch = chunked_docs[i : i + batch_size]
            vectorstore.add_documents(batch)
            total_loaded += len(batch)
            logger.info(f"   Loaded batch {i // batch_size + 1}: {total_loaded}/{len(chunked_docs)} documents")

        logger.info(f"Successfully loaded {total_loaded} document chunks into Qdrant vector store.")

    except Exception as e:
        logger.error(f"Failed to load documents: {e}")
        raise ValueError(f"Failed to load documents: {e}")
    
    stats = {
        "total_documents": len(documents),
        "total_chunks": len(chunked_docs),
        "avg_chunk_size": sum(len(c.page_content) for c in chunked_docs) // len(chunked_docs),
        "collection_name": "production_documents",
        "embedding_model": "text-embedding-3-small",
        "embedding_dimensions": 1536,
    }
    return stats

if __name__ == "__main__":
    build_and_load_knowledge_base()