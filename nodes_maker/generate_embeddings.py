import os
import logging
from typing import List, Tuple
from dotenv import load_dotenv
from litellm import embedding, EmbeddingResponse
from sqlmodel import Session, select
from utils.db import engine
from utils.schema import (
    Node as DBNode,
    ContentData,
    Embedding,
    EmbeddingSource
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))  # Adjust based on input sizes and provider

def iter_content_without_embeddings(session: Session, document_ids: list[int] | None = None, limit: int | None = None):
    # Base filters for content that actually needs embeddings
    needs_embedding_filter = (
        (Embedding.id.is_(None)) &
        (
            (
                (ContentData.embedding_source == EmbeddingSource.TEXT_CONTENT) &
                (ContentData.text_content.is_not(None))
            ) |
            (
                (ContentData.embedding_source == EmbeddingSource.DESCRIPTION) &
                (ContentData.description.is_not(None))
            ) |
            (
                (ContentData.embedding_source == EmbeddingSource.CAPTION) &
                (ContentData.caption.is_not(None))
            )
        )
    )

    # Step 1: choose up to `limit` document_ids that have pending content
    docs_stmt = (
        select(DBNode.document_id)
        .join(ContentData, ContentData.node_id == DBNode.id)
        .outerjoin(Embedding, Embedding.content_data_id == ContentData.id)
        .where(needs_embedding_filter)
        .group_by(DBNode.document_id)
    )
    if document_ids:
        docs_stmt = docs_stmt.where(DBNode.document_id.in_(document_ids))
    if limit is not None:
        docs_stmt = docs_stmt.limit(limit)
    docs_subq = docs_stmt.subquery()

    # Step 2: fetch all ContentData in those documents that still need embeddings
    content_stmt = (
        select(ContentData)
        .join(DBNode, ContentData.node_id == DBNode.id)
        .outerjoin(Embedding, Embedding.content_data_id == ContentData.id)
        .where(needs_embedding_filter)
        .where(DBNode.document_id.in_(select(docs_subq.c.document_id)))
    )

    return session.exec(content_stmt)


def generate_embeddings_batch(texts: List[str], api_key: str) -> Tuple[List[List[float]], str]:
    """Generate embedding vectors for a batch of texts and return (vectors, model_name)."""
    response: EmbeddingResponse = embedding(model=DEFAULT_EMBEDDING_MODEL, input=texts, api_key=api_key)
    # Sort by index to preserve input order
    vectors: List[List[float]] = [item["embedding"] for item in sorted(response.data, key=lambda d: d["index"])]
    model_name: str = response.model if response.model else DEFAULT_EMBEDDING_MODEL
    return vectors, model_name


def batch_items(items: List, size: int) -> List[List]:
    """Split a list into batches of specified size."""
    return [items[i:i+size] for i in range(0, len(items), size)]


def generate_embeddings(document_ids: list[int] | None = None, limit: int | None = None, api_key: str | None = None) -> None:
    """Generate and persist embeddings for all content without embeddings."""
    with Session(engine) as session:
        # Collect all content items that need embeddings
        content_items = list(iter_content_without_embeddings(session, document_ids, limit))
        
        if not content_items:
            logger.info("No content items need embeddings")
            return
        
        # Process in batches
        for batch in batch_items(content_items, BATCH_SIZE):
            # Extract texts from content items
            texts = []
            for cd in batch:
                if cd.embedding_source == EmbeddingSource.TEXT_CONTENT:
                    texts.append(cd.text_content)
                elif cd.embedding_source == EmbeddingSource.DESCRIPTION:
                    texts.append(cd.description)
                else:
                    texts.append(cd.caption)
            
            # Generate embeddings for the batch
            vectors, model_name = generate_embeddings_batch(texts, api_key)

            # Create Embedding objects maintaining one-to-one relationship
            for cd, vector in zip(batch, vectors):
                session.add(Embedding(content_data_id=cd.id, embedding_vector=vector, model_name=model_name))
            
            logger.info(f"Generated embeddings for batch of {len(batch)} items")
        
        session.commit()
        logger.info(f"Successfully generated embeddings for {len(content_items)} content items")


if __name__ == "__main__":
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY is not set"
    
    generate_embeddings(limit=1, api_key=api_key)