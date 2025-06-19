# Revised Schema Design

We are going to have an LLM turn the PDF into an HTML document, and then we will programmatically convert the DOM to a graph for storage in (and retrieval from) our database. Our schema will closely follow the data model of the DOM (for ease of ingestion and reconstruction), but we will enrich it with additional fields (provided as `data-` attributes on the DOM nodes in the HTML) and relationships (created from anchor links in the HTML).

As in the DOM, we will have an enum to represent whether the node is an element or a text node. We will capture top-level document structure-- front matter, body matter, and back matter boundaries-- with `header`, `main`, and `footer` tags, respectively. ToC-type sections will be represented by `nav` tags. Notes, sidebars, and text boxes will be `aside` elements. We'll make liberal use of `section` tags to represent the document's structure, with `data-section-type` attributes to capture richer semantic labels for the section type.

We'll exclude page headers and footers in our HTML. Each HTML element will have `data-start-page` and (optionally, if the element spans multiple pages) `data-end-page` attributes, which will use PDF page numbers. We'll have an LLM map PDF page numbers to logical page numbers and use that to enrich the positional data. We'll also likely use cosine similarity to map elements to PDF text blocks to extract the bounding box for enriching our database records with positional data.

We must limit the tags available to the LLM to those we want to support, e.g., `header`, `main`, `footer`, `figure`, `figcaption`, `table`, `th`, `tr`, `td`, `caption`, `title`, `section`, `nav`, `aside`, `p`, `ul`, `ol`, `li`, `h1`, `h2`, `h3`, `h4`, `h5`, `h6`, `i`, `b`, `u`, `s`, `sup`, `sub`, `a`, `img`, `svg`, `math`, `code`, `cite`, `blockquote`.

For `TEXT_NODE`s and `img`-typed `ELEMENT_NODE`s, we will have a corresponding record in a content data table that will contain the text content or the image URL, respectively.

```mermaid
erDiagram
    %% ENUM: NodeType
    NodeType {
        string TEXT_NODE
        string ELEMENT_NODE
    }

    %% ENUM: TagName
    TagName {
        string HEADER
        string MAIN
        string FOOTER
        string FIGURE
        string FIGCAPTION
        string TABLE
        string TH
        string TR
        string TD
        string CAPTION
        string TITLE
        string SECTION
        string NAV
        string ASIDE
        string P
        string UL
        string OL
        string LI
        string H1
        string H2
        string H3
        string H4
        string H5
        string H6
        string A
        string IMG
        string SVG
        string MATH
        string CODE
        string CITE
        string BLOCKQUOTE
    }

    %% ENUM: SectionType
    SectionType {
        string ABSTRACT
        string ACKNOWLEDGEMENTS
        string APPENDIX
        string BIBLIOGRAPHY
        string CHAPTER
        string CONCLUSION
        string COPYRIGHT_PAGE
        string DEDICATION
        string EPILOGUE
        string EXECUTIVE_SUMMARY
        string FOOTER
        string FOREWORD
        string HEADER
        string INDEX
        string INTRODUCTION
        string LIST_OF_BOXES
        string LIST_OF_FIGURES
        string LIST_OF_TABLES
        string NOTES_SECTION
        string PART
        string PREFACE
        string PROLOGUE
        string SECTION
        string STANZA
        string SUBSECTION
        string TABLE_OF_CONTENTS
        string TEXT_BOX
        string TITLE_PAGE
    }

    %% ENUM: EmbeddingSource
    EmbeddingSource {
        string TEXT_CONTENT "Embed the primary text content"
        string DESCRIPTION  "Embed the AI-generated description (for tables, figures)"
        string CAPTION "Embed the original caption (for figures, tables)"
    }

    %% Unified Node Structure
    NODE {
        int id PK
        int document_id FK
        NodeType node_type
        TagName tag_name nullable "The HTML tag name of the node if it is an element node"
        SectionType section_type nullable "The semantic section type of the node if it is a section element"
        int parent_id FK nullable "The ID of the parent node"
        int sequence_in_parent "The sequence number of the node within its parent"
        jsonb positional_data "[{page_pdf: int, page_logical: int, bbox: {x1: float, y1: float, x2: float, y2: float}}, ...]" "JSONB array of positional data for the PDF blocks that make up the node"
    }

    %% Content Data (1:1 with content-bearing nodes)
    CONTENT_DATA {
        int id PK
        int node_id FK
        text text_content nullable
        string storage_url nullable
        string description nullable
        string caption nullable
        EmbeddingSource embedding_source
    }

    %% ENUM: RelationType (for non-hierarchical links)
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
    }
```
