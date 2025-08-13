import os
from typing import List, Tuple
from dotenv import load_dotenv
from litellm import embedding, EmbeddingResponse
from sqlmodel import Session, select, and_, or_
from utils.db import engine
from utils.schema import (
    Node as DBNode,
    ContentData,
    Embedding,
    EmbeddingSource
)

load_dotenv()

DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

def iter_content_without_embeddings(session, document_ids: list[int] | None = None):
    stmt = (
        select(ContentData)
        .join(DBNode, ContentData.node_id == DBNode.id)
        .outerjoin(Embedding, Embedding.content_data_id == ContentData.id)
        .where(Embedding.id.is_(None))
    )

    if document_ids:
        stmt = stmt.where(DBNode.document_id.in_(document_ids))

    # Optional: filter to only rows that actually have text to embed
    stmt = stmt.where(
        or_(
            and_(
                ContentData.embedding_source == EmbeddingSource.TEXT_CONTENT,
                ContentData.text_content.is_not(None),
            ),
            and_(
                ContentData.embedding_source == EmbeddingSource.DESCRIPTION,
                ContentData.description.is_not(None),
            ),
            and_(
                ContentData.embedding_source == EmbeddingSource.CAPTION,
                ContentData.caption.is_not(None),
            ),
        )
    )

    return session.exec(stmt)


def generate_embedding(text: str) -> Tuple[List[float], str]:
    """Generate an embedding vector and return (vector, model_name)."""
    response: EmbeddingResponse = embedding(model=DEFAULT_EMBEDDING_MODEL, input=text)
    vector: List[float] = response.data[0].embedding  # type: ignore[assignment]
    model_name: str = response.model  # type: ignore[assignment]
    return vector, model_name


def generate_embeddings(document_ids: list[int] | None = None) -> None:
    """Generate and persist embeddings for all content without embeddings."""
    with Session(engine) as session:
        for cd in iter_content_without_embeddings(session, document_ids):
            if cd.embedding_source == EmbeddingSource.TEXT_CONTENT:
                text = cd.text_content
            elif cd.embedding_source == EmbeddingSource.DESCRIPTION:
                text = cd.description
            else:
                text = cd.caption

            vector, model_name = generate_embedding(text)
            session.add(Embedding(content_data_id=cd.id, embedding_vector=vector, model_name=model_name))
        session.commit()


if __name__ == "__main__":
    generate_embeddings()