from langchain_core.tools import tool, Tool

from app.monitoring import get_logger
from app.hybrid_search import hybrid_search


logger = get_logger()

def create_retriever_tool(
    top_k: int | None = None, 
    semantic_weight: float | None = None
) -> Tool:
    """Create a retriever tool.

    Args:
        query: Search query
        top_k: Number of documents to return (default: 5)
        semantic_weight: Weight of vector search (default: 6)

    """

    @tool
    async def retrieve_answer(query: str) -> str:
        """Search the knowledge base and retrieve relevant HR context.

        Args:
            query: The search query describing what papers to find
        
        Returns:
            str: Text with all retrieved context
        """

        logger.info(f"Retrieving documents for query: {query[:100]}...")
    
        docs = await hybrid_search(query, top_k, semantic_weight)

        return "\n\n".join([f"Document {i+1}: \n{doc.page_content}. (Source: {doc.metadata.get('source', '')})" for i, doc in enumerate(docs)])

    return retrieve_answer