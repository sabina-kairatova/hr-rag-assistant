from langchain_openai import OpenAIEmbeddings
from app.config import get_settings


def create_embeddings() -> OpenAIEmbeddings:
    """Create OpenAI embeddings instance for RAG."""

    settings = get_settings()

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key
    )