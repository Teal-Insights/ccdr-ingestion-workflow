from utils.models import StructuredNode
from sqlmodel import Session
from utils.db import engine
from utils.schema import Node as DBNode, ContentData, EmbeddingSource, TagName

def upload_structured_nodes_to_db(nested_structure: list[StructuredNode], document_id: int) -> None:
    with Session(engine) as session:
        def _upload(node: StructuredNode, parent_id: int | None, seq: int):
            db_node = DBNode(
                document_id=int(document_id),
                tag_name=node.tag,
                section_type=node.section_type,
                parent_id=parent_id,
                sequence_in_parent=seq,
                positional_data=[pos.dict() for pos in node.positional_data],
            )
            session.add(db_node)
            session.flush()

            # Create ContentData for text or image nodes
            if node.tag == TagName.IMG or node.text is not None:
                embedding_source = EmbeddingSource.DESCRIPTION if node.tag == TagName.IMG else EmbeddingSource.TEXT_CONTENT
                text_content = node.text if node.tag != TagName.IMG else None

                content_data = ContentData(
                    node_id=db_node.id,
                    text_content=text_content,
                    storage_url=node.storage_url,
                    description=node.description,
                    caption=node.caption,
                    embedding_source=embedding_source,
                )
                session.add(content_data)

            # Recursively upload child nodes
            for idx, child in enumerate(node.children):
                _upload(child, db_node.id, idx)

        for idx, root in enumerate(nested_structure):
            _upload(root, None, idx)
        session.commit()

if __name__ == "__main__":
    import os
    import json
    from utils.models import StructuredNode

    # Load nested structure from JSON (for testing purposes)
    file_path = os.path.join("artifacts", "doc_601_nested_structure_classified.json")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Parse into StructuredNode objects
    nested_structure = [StructuredNode.model_validate(item) for item in data]

    # Test document_id (replace with actual ID as needed)
    test_document_id = 601
    upload_structured_nodes_to_db(nested_structure, test_document_id)
    print(f"Uploaded nested structure to DB for document_id {test_document_id}")