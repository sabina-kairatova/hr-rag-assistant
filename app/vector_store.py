from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import get_settings

# from config import get_settings


def get_qdrant_client() -> QdrantClient:
    """Initialize Qdrant client with environment-aware configuration.

    Returns:
        QdrantClient: Configured Qdrant client instance
    """
    settings = get_settings()

    qdrant_url = settings.qdrant_url
    api_key = settings.qdrant_api_key

    if api_key:
        return QdrantClient(url=qdrant_url, api_key=api_key)
    else:
        return QdrantClient(url=qdrant_url)


def create_collection_if_not_exists(
        client: QdrantClient,
        collection_name: str | None = None,
        vector_size: int = 1536,
        Distance: Distance = Distance.COSINE,
    ) -> None:
    """Create Qdrant collection if it doesn't exist.

    Args:
        client: Qdrant client instance
        collection_name: Name of the collection (default: weather_knowledge)
        vector_size: Embedding dimension (default: 1536 for text-embedding-3-small)
        distance: Distance metric (default: COSINE for OpenAI embeddings)
    """

    collection_name = get_settings().collection_name if collection_name is None else collection_name

    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance
            ),
        )

def get_vector_store(embeddings: OpenAIEmbeddings = None) -> QdrantVectorStore:
    """Get configured Qdrant vector store for RAG.
    
    Returns:
        QdrantVectorStore: Configured Qdrant vectore store instance
    """

    if embeddings is None:
        from app.embeddings import create_embeddings
        embeddings = create_embeddings()

    client = get_qdrant_client()

    create_collection_if_not_exists(client)

    return QdrantVectorStore(
        client=client,
        embedding=embeddings,
        collection_name=get_settings().collection_name,
    )

