# CCDR Ingestion Workflow

A complete workflow to transform World Bank Country and Climate Development Reports (CCDRs) from PDF format into a structured graph database format suitable for semantic search and retrieval.

## Overview

This project processes PDF documents from PostgreSQL/AWS S3 storage through a multi-stage pipeline that extracts layout information, images, and text content, then uses Large Language Models (LLMs) to structure the content hierarchically as HTML DOM before uploading structured nodes back to the database. [See here](https://github.com/Teal-Insights/ccdr-explorer-api/blob/main/schema.md) for the database schema and discussion of schema design.

## Architecture

The pipeline consists of 11 main stages:

1. **Document Discovery** - Identifies unprocessed documents from the database
2. **PDF Acquisition** - Downloads PDFs from S3 or directly from World Bank URLs  
3. **Layout Analysis** - Extracts bounding boxes and element labels using Layout Extractor API
4. **Logical Page Mapping** - Maps physical pages to logical page numbers using LLM analysis
5. **Content Block Reclassification** - Reclassifies content blocks to improve accuracy
6. **Image Extraction** - Extracts images using PyMuPDF
7. **Image Description** - Describes images using Vision Language Models
8. **Text Styling** - Applies formatting information from PDF to text blocks
9. **Top-Level Structure Detection** - Identifies front, body, and back matter using LLM analysis
10. **Nested Structure Detection** - Converts the top-level structure into a nested HTML DOM structure using LLM analysis
11. **Database Ingestion** - Converts structured content to database directed graph nodes and uploads to PostgreSQL
12. **Relation Enrichment** - Generate relationships from anchor tags and references (TODO)
13. **Embedding Generation** - Generate embeddings for each ContentData record (TODO)

## Schema Evolution

The project has evolved from a complex multi-stage schema (see `schema_legacy.md`) to a simplified DOM-based approach (see `schema_revision.md`). The current schema closely follows HTML DOM structure while adding semantic enrichments through data attributes and relationships.

### Current Schema Features

- **DOM-based Structure**: Mirrors HTML element hierarchy for easy reconstruction
- **Semantic Enrichment**: Uses `data-section-type` attributes for rich semantic labeling
- **Positional Data**: Maintains PDF page numbers and bounding boxes
- **Content Relationships**: Captures citations, footnotes, and cross-references
- **Multi-modal Support**: Handles text, images, and vector graphics uniformly

## Installation

This project uses `uv` for dependency management:

```bash
# Install dependencies
uv sync

# Or add new dependencies
uv add package_name
```

## Configuration

Create a `.env` file:

```
# LLM API credentials
GEMINI_API_KEY=
DEEPSEEK_API_KEY=
OPENROUTER_API_KEY=
OPENAI_API_KEY=

# Temporary bug fix to prevent litellm resource leakage
DISABLE_AIOHTTP_TRANSPORT=True

# Experimental ML document layout extraction service
LAYOUT_EXTRACTOR_API_URL=
LAYOUT_EXTRACTOR_API_KEY=

# AWS S3 credentials
S3_BUCKET_NAME=
AWS_REGION=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=

# Database instance
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=ccdr-explorer-db
```

## Usage

### Full Pipeline

Run the complete pipeline to process unprocessed documents from the database:

```bash
uv run ingest_ccdrs.py
```

This will:
1. Query the database for unprocessed documents (documents with no child nodes)
2. Download PDFs from S3 or World Bank URLs
3. Process documents through the complete pipeline
4. Upload structured nodes back to the database
5. Output all intermediate artifacts to `./artifacts/wkdir` for debugging

The pipeline processes documents in batches (configurable via `LIMIT` variable) and includes comprehensive error handling with fail-fast validation for required environment variables and database schema synchronization.

### Individual Components

You can also test individual transformation components:

#### Layout Extraction
```bash
uv run -m transform.extract_layout document.pdf output.json
```

#### Page Number Mapping
```bash
uv run -m transform.map_page_numbers
```

#### Image Description
```bash
uv run -m transform.describe_images
```

#### Structure Detection
```bash
uv run -m transform.detect_structure
```

## Project Structure

```
ccdr-ingestion-workflow/
├── ingest_ccdrs.py            # Main pipeline orchestrator
├── transform/                 # Core transformation modules
│   ├── extract_layout.py      # PDF layout extraction using Layout Extractor API
│   ├── map_page_numbers.py    # Logical page number mapping using LLM router
│   ├── reclassify_blocks.py   # Content block type reclassification
│   ├── extract_images.py      # Image extraction from PDF
│   ├── describe_images.py     # Image description using Vision Language Models
│   ├── style_text_blocks.py   # Text styling from PDF formatting
│   ├── detect_top_level_structure.py # Top-level document structure detection
│   ├── detect_structure.py    # Nested structure detection with concurrency control
│   ├── upload_to_db.py        # Database upload functionality
│   └── models.py              # Pydantic data models
├── utils/                     # Utility modules
│   ├── db.py                  # Database connection and schema validation
│   ├── schema.py              # Database schema definitions
│   ├── aws.py                 # S3 and AWS operations
│   └── html.py                # HTML processing utilities
├── artifacts/                 # Working directory for pipeline outputs
└── pyproject.toml            # Project dependencies
```

## Key Features

### Database-Driven Processing
- Queries PostgreSQL database for unprocessed documents
- Downloads PDFs from S3 or World Bank URLs as fallback
- Uploads structured content back to database as graph nodes
- Fail-fast validation for environment variables and schema sync

### Advanced Layout Analysis
- Uses dedicated Layout Extractor API for precise bounding box detection
- Intelligent logical page number mapping using LLM analysis
- Content block reclassification to improve accuracy
- Header/footer filtering based on logical page analysis

### Multi-modal Content Processing
- **Text**: Preserves styling and formatting from original PDF
- **Images**: Automatic extraction with context-aware AI descriptions
- **Structure**: Hierarchical document organization with nested sections

### Intelligent Structure Detection
- Two-stage structure detection (top-level and nested)
- Uses multiple LLM providers with router-based load balancing
- Concurrency control for efficient API usage
- Context-aware section identification

### Robust API Integration
- LiteLLM Router with advanced load balancing and fallbacks
- Multiple provider support (Gemini, OpenAI, DeepSeek, OpenRouter)
- Built-in retry logic and error handling
- Configurable rate limiting and concurrency control

### Scalable Processing
- Async/await patterns throughout the pipeline
- Batch processing with configurable limits
- Comprehensive error handling and recovery
- Intermediate artifact preservation for debugging

## Dependencies

- **SQLModel**: Database ORM and schema definitions
- **PostgreSQL**: Primary database for document storage
- **PyMuPDF**: PDF parsing and content extraction
- **Pillow**: Image processing and conversion
- **LiteLLM**: Unified LLM API interface with router support
- **Pydantic**: Data validation and serialization
- **Tenacity**: Retry logic for API calls
- **Boto3**: AWS S3 integration for PDF storage

## API Usage

The project uses multiple LLM providers through LiteLLM Router:
- **Gemini**: Image description and top-level structure detection
- **OpenAI**: Available through router for various tasks
- **DeepSeek**: Page number mapping and nested structure detection
- **OpenRouter**: Alternative provider access

The router provides load balancing, fallbacks, and automatic retry logic with configurable concurrency limits.

## Development Status

### Completed
- ✅ Database-driven document discovery and processing
- ✅ PDF download from S3 and World Bank URLs
- ✅ Layout extraction using dedicated API
- ✅ Logical page number mapping with LLM analysis
- ✅ Content block reclassification and filtering
- ✅ Image extraction and description with VLM
- ✅ Text styling preservation from PDF formatting
- ✅ Two-stage hierarchical structure detection
- ✅ Database ingestion of structured content
- ✅ Concurrent processing with semaphore control
- ✅ LiteLLM Router integration with multiple providers

### In Progress
- 🚧 Relationship extraction from anchor tags (`create_relations.py`)
- 🚧 Vector embedding generation for semantic search

### Planned
- 📋 Enhanced error recovery
- 📋 Performance optimization and batch size tuning
- 📋 Enhanced LLM response validation/evaluation
- 📋 Fine-tuning dataset prep and/or similar-example injection

## Contributing

This project uses modern Python practices:
- Type hints throughout
- Pydantic models for data validation
- Async/await for concurrent operations
- Comprehensive error handling and retry logic

## License

MIT

## Acknowledgments

This project processes World Bank Country and Climate Development Reports (CCDRs) to make them more accessible for research and analysis through semantic search and retrieval systems.
