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


def generate_embedding(text: str, api_key: str) -> Tuple[List[float], str]:
    """Generate an embedding vector and return (vector, model_name)."""
    response: EmbeddingResponse = embedding(model=DEFAULT_EMBEDDING_MODEL, input=text, api_key=api_key)
    assert len(response.data) == 1, "Expected a single embedding; got a batch."
    vector: List[float] = response.data[0]["embedding"]
    if response.model:
        model_name: str = response.model
    else:
        model_name = DEFAULT_EMBEDDING_MODEL
    return vector, model_name


def generate_embeddings(document_ids: list[int] | None = None, limit: int | None = None, api_key: str | None = None) -> None:
    """Generate and persist embeddings for all content without embeddings."""
    with Session(engine) as session:
        for cd in iter_content_without_embeddings(session, document_ids, limit):
            if cd.embedding_source == EmbeddingSource.TEXT_CONTENT:
                text = cd.text_content
            elif cd.embedding_source == EmbeddingSource.DESCRIPTION:
                text = cd.description
            else:
                text = cd.caption

            vector, model_name = generate_embedding(text, api_key)
            session.add(Embedding(content_data_id=cd.id, embedding_vector=vector, model_name=model_name))
        session.commit()


if __name__ == "__main__":
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY is not set"
    
    generate_embeddings(limit=1, api_key=api_key)