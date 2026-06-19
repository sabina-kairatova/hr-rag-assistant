from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.vector_store import get_qdrant_client
from app.config import get_settings
from app.monitoring import get_logger
from app.vector_store import get_vector_store

logger = get_logger()

def get_all_docs_from_vectorstore() -> list[Document]:
    """Retrieve all documents from Qdrant vector store for BM25 indexing.
    
    Returns:
        list[Document]: All documents in the vector store
    """
    clinet = get_qdrant_client()
    collection_name = get_settings().collection_name

    all_docs = []
    offset = None
    batch_size=100

    while True:
        result = clinet.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        points, next_offset = result

        if not points:
            break

        docs = [
            Document(page_content=point.payload.get("page_content", ""), 
                     metadata=point.payload.get("metadata", {})) for point in points
        ]
        all_docs.extend(docs)

        if next_offset is None:
            break

        offset = next_offset
        
    logger.info(f"✅ Loaded {len(all_docs)} documents from Qdrant for BM25 indexing")
    return all_docs

class HybridSearchManager:    
    """Class-based search management"""
    
    def __init__(self):
        self._vector_retriever = None
        self._bm25_retriever = None
        self._ensemble_retriever = None
        self._initialized = False
    
    def _initialize_retrievers(self, top_k: int, semantic_weight: float) -> None:
        if not self._initialized:
            bm25_weight = 1 - semantic_weight
            vectorstore = get_vector_store()
            self._vector_retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k}
            )
            logger.info("✅ Vector retriever initialized.", extra={"extra_data": {
                "search_type": "similarity", "k": top_k
            }})

            all_docs = get_all_docs_from_vectorstore()
            self._bm25_retriever = BM25Retriever.from_documents(all_docs)
            self._bm25_retriever.k = top_k
            logger.info("✅ BM25 retriever: initialized.", extra={"extra_data": {
                "k": top_k, "len_indexed_data": len(all_docs)
            }})

            self._ensemble_retriever = EnsembleRetriever(
                retrievers=[self._vector_retriever, self._bm25_retriever],
                weight=[semantic_weight, bm25_weight]
            )
            logger.info("✅ Ensemble retriever: initialized.", extra={"extra_data": {
                "semantic_weight": semantic_weight, "bm25_weight": bm25_weight
            }})
            self._initialized = True

    def get_hybrid_retriever(
            self, top_k: int | None = None, semantic_weight: float | None = None
        ) -> EnsembleRetriever:
        
        if top_k is None:
            top_k = get_settings().default_k
        if semantic_weight is None:
            semantic_weight = get_settings().semantic_weight

        self._initialize_retrievers(top_k=top_k, semantic_weight=semantic_weight)
        return self._ensemble_retriever

async def hybrid_search(
        query: str, top_k: int | None = None, semantic_weight: float | None = None
    ) -> list[Document]:
    
    """Perform hybrid search with Ensembel Retrieval.
    Args:
        query: Search query
        top_k: Number of documents to return (default: 5)
        semantic_weight: Weight of vector search (default: 6)

    Returns:
        list[Document]: Retrieved documents ranked by hybrid score
    """

    manager = HybridSearchManager()
    hybrid_retriever = manager.get_hybrid_retriever(top_k, semantic_weight)

    docs = await hybrid_retriever.ainvoke(input=query)

    logger.info(f"🔍 Hybrid search complete.", extra={"extra_data": {
        "query": query[:50] + '...', "len_retrieved_docs": len(docs)
    }})
    return docs
