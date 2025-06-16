# PDF Parsing Eval

In this repository, I will evaluate the performance of different tools and workflows for parsing PDFs for RAG.

I have a particular schema I want to coerce the data into. Eventually I will likely train a model to convert PDFs to this schema in a single step, but for now I expect to need a multi-step workflow to achieve what I want.

I will first create some ideal human-prepared data for a few PDF pages (extracted with `qpdf dl_001.pdf --pages dl_001.pdf --range=1-5 -- output.pdf`), and then I will use cosine similarity on outputs from various open-source and commercial tools to see which ones are best suited for my use case.

## Schema

The master schema below represents the final, normalized data model we aim to produce. The data is generated through a multi-stage process detailed in the following section.

```mermaid
erDiagram
    %% Relationship lines
    DOCUMENT_COMPONENT ||--o{ DOCUMENT_COMPONENT : "contains (self-reference)"
    DOCUMENT_COMPONENT ||--o{ CONTENT_NODE : contains
    CONTENT_NODE      ||--o{ RELATION : source_of
    CONTENT_NODE      ||--o{ RELATION : target_of

    %% ENUM: ComponentType (Structural Nodes)
    ComponentType {
        %% Top-level components
        string FRONT_MATTER
        string BODY_MATTER
        string BACK_MATTER

        %% Generic sub-components
        string CONTAINER
        string SECTION
        string LIST

        %% Containers
        string COPYRIGHT_PAGE
        string FOOTER
        string HEADER
        string TEXT_BOX
        string TITLE_PAGE

        %% Lists
        string BIBLIOGRAPHY
        string LIST_OF_BOXES
        string LIST_OF_TABLES
        string LIST_OF_FIGURES
        string NOTES_SECTION
        string TABLE_OF_CONTENTS

        %% Sections
        string ABSTRACT
        string ACKNOWLEDGEMENTS
        string APPENDIX
        string CHAPTER
        string CONCLUSION
        string DEDICATION
        string EPILOGUE
        string EXECUTIVE_SUMMARY
        string FOREWORD
        string INDEX
        string INTRODUCTION
        string PART
        string PREFACE
    }

    %% ENUM: ContentNodeType (Content Nodes)
    ContentNodeType {
        string AUTHOR
        string BLOCK_QUOTATION
        string BIBLIOGRAPHIC_ENTRY
        string CAPTION
        string FIGURE
        string FORMULA
        string HEADING
        string LIST_ITEM
        string NOTE
        string PARAGRAPH
        string PAGE_NUMBER
        string STANZA
        string SUBHEADING
        string SUBTITLE
        string TABLE
        string TITLE
    }

    %% ENTITY: DOCUMENT_COMPONENT (The Containers/Structure)
    DOCUMENT_COMPONENT {
        int id PK
        string document_id FK
        ComponentType component_type "The type of structural container"
        string title "The heading/title of this component, e.g., 'Chapter 1: Introduction'"
        int parent_component_id FK "Self-referencing FK to build the hierarchy"
        int sequence_in_parent "Order of this component within its parent"
        int4range page_range "Page range of the component (inclusive)"
    }

    %% ENUM: EmbeddingSource
    EmbeddingSource {
        string TEXT_CONTENT "Embed the primary text content"
        string DESCRIPTION  "Embed the AI-generated description (for tables, figures)"
        string CAPTION "Embed the original caption (for figures, tables)"
    }

    %% ENTITY: CONTENT_NODE
    CONTENT_NODE {
        int id PK
        string document_id FK
        int parent_component_id FK "FK to the DOCUMENT_COMPONENT that contains this node"
        ContentNodeType content_node_type
        text content "The primary, cleaned text content of the node"
        string storage_url "For binary content like images"
        string description "AI-generated summary/description (for figures, tables)"
        EmbeddingSource embedding_source "Which field to use for the vector embedding"
        int sequence_in_parent_major "Order of this chunk within its parent component"
        int sequence_in_parent_minor "Zero unless the node is a footnote or sidebar, in which case it indicates reading order among these supplementary nodes"
        jsonb positional_data "[{page_pdf: int, page_logical: int, bbox: {x1: float, y1: float, x2: float, y2: float}}, ...]"
    }

    %% ENUM: RelationType (For non-hierarchical links)
    RelationType {
        string REFERENCES_NOTE "Text references a footnote or endnote"
        string REFERENCES_CITATION "Text references a bibliographic entry"
        string IS_CAPTIONED_BY "A node is a caption for another node"
        string IS_SUPPLEMENTED_BY "A node is supplemented by another node (e.g., a sidebar or legend)"
        string CONTINUES "A node continues from a previous one (e.g., across sections)"
        string CROSS_REFERENCES "A node references another arbitrary node"
    }

    %% ENTITY: RELATION
    RELATION {
        int id PK "Unique relation identifier"
        int source_node_id FK "The origin node of the relationship"
        int target_node_id FK "The destination node of the relationship"
        RelationType relation_type
        string marker_text "Optional text for the relation, e.g., '1' for a footnote or '(Author, 2025)' for a citation"
    }

    %% ===== CSS STYLING =====
    classDef enumType fill:#ffe6e6,stroke:#ff4757
    classDef mainTable fill:#e6f3ff,stroke:#0066cc

    class ComponentType,RelationType,ContentNodeType,EmbeddingSource enumType
    class DOCUMENT_COMPONENT,CONTENT_NODE,RELATION mainTable
```

## Staged Data Generation Workflow

We use a three-stage workflow to transform a raw PDF into the structured data defined in the master schema. Each stage builds upon the last, progressively adding detail and structure.

### Stage 1: Hierarchical Component Generation

In the first stage, an LLM analyzes the PDF to identify the main structural components and their hierarchical relationships. The output is a nested structure that captures the document's outline.

```mermaid
erDiagram
    DOCUMENT_COMPONENT ||--o{ DOCUMENT_COMPONENT : "nests"

    DOCUMENT_COMPONENT {
        ComponentType component_type
        string title nullable
        DOCUMENT_COMPONENT[] children
        int4range page_range "Two-element array representing the inclusive page range of the component, e.g., [1, 10]"
    }

    %% ===== CSS STYLING =====
    classDef enumType fill:#ffe6e6,stroke:#ff4757
    classDef mainTable fill:#e6f3ff,stroke:#0066cc

    class ComponentType enumType
    class DOCUMENT_COMPONENT mainTable
```

### Stage 2: Content Node Mapping

The nested structure from Stage 1 is programmatically flattened, and temporary IDs are assigned to each component. A second LLM pass then reads the PDF content and maps text, figures, tables, and other content nodes to their parent components using these IDs.

```mermaid
erDiagram
    DOCUMENT_COMPONENT ||--o{ CONTENT_NODE : "contains"

    DOCUMENT_COMPONENT {
        int id PK
        string title
        ComponentType component_type
        int4range page_range
    }

    CONTENT_NODE {
        int parent_component_id FK
        ContentNodeType content_node_type
        text content
        int sequence_in_parent_minor
    }

    %% ===== CSS STYLING =====
    classDef enumType fill:#ffe6e6,stroke:#ff4757
    classDef mainTable fill:#e6f3ff,stroke:#0066cc

    class ComponentType,ContentNodeType enumType
    class DOCUMENT_COMPONENT,CONTENT_NODE mainTable
```

### Stage 3: Relation Annotation

Finally, temporary IDs are generated for all content nodes. A third LLM pass identifies non-hierarchical relationships between these nodes, such as footnote references, citations, and cross-references, creating the final `RELATION` entities.

```mermaid
erDiagram
    CONTENT_NODE ||--o{ RELATION : "source_of"
    CONTENT_NODE ||--o{ RELATION : "target_of"

    CONTENT_NODE {
        int id PK
        ContentNodeType content_node_type
        text content
    }

    RELATION {
        int id PK
        int source_node_id FK
        int target_node_id FK
        RelationType relation_type
        string marker_text
    }

    %% ===== CSS STYLING =====
    classDef enumType fill:#ffe6e6,stroke:#ff4757
    classDef mainTable fill:#e6f3ff,stroke:#0066cc

    class RelationType,ContentNodeType enumType
    class CONTENT_NODE,RELATION mainTable
```
