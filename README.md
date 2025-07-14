# CCDR Ingestion Workflow

A complete workflow to transform World Bank Country and Climate Development Reports (CCDRs) from PDF format into a RAG-friendly schema and upload them to PostgreSQL for semantic search and retrieval.

## Overview

This project processes PDF documents through a multi-stage pipeline that extracts text, images, and vector graphics, then uses Large Language Models (LLMs) to clean and structure the content into semantic HTML before converting it to a graph database schema optimized for Retrieval-Augmented Generation (RAG) applications. [See here](https://github.com/Teal-Insights/ccdr-explorer-api/blob/main/schema.md) for the database schema and discussion of schema design.

## Architecture

The pipeline consists of 10 main stages:

1. **Text Block Extraction** - Extract text blocks with styling and positioning from PDF
2. **Image Extraction** - Extract and describe images using vision-language models
3. **SVG Extraction** - Extract vector graphics and generate descriptions
4. **Block Combination** - Merge all extracted blocks into a unified document
5. **HTML Conversion** - Convert blocks to structured HTML with semantic IDs
6. **Structure Detection** - Use LLM to identify document sections (header/main/footer)
7. **Rich HTML Generation** - Create styled HTML with positioning data
8. **HTML Cleaning** - LLM-based cleaning to conform to semantic specification
9. **Graph Conversion** - Transform HTML DOM to database graph schema
10. **Relation Enrichment** - Generate relationships from anchor tags and references

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

Set up your environment variables:

```bash
# Required API keys
export GEMINI_API_KEY="your_gemini_api_key"
export DEEPSEEK_API_KEY="your_deepseek_api_key"
```

Or create a `.env` file:

```
GEMINI_API_KEY=your_gemini_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
```

## Usage

### Full Pipeline

Run the complete pipeline on a PDF:

```bash
# Place your PDF as 'input.pdf' in the project root
uv run ingest_ccdrs.py
```

This will process the PDF through all stages and output the results to a temporary directory.

### Individual Stages

You can also run individual components:

#### Extract Text Blocks
```bash
uv run -m transform.extract_text_blocks document.pdf
```

#### Extract Images
```bash
uv run -m transform.extract_images document.pdf
```

#### Extract SVGs
```bash
uv run -m transform.extract_svgs document.pdf
```

#### Combine Blocks
```bash
uv run -m transform.combine_blocks output.json text_blocks.json images.json svgs.json
```

#### Convert to HTML
```bash
uv run -m transform.convert_to_html combined_blocks.json output.html --rich-text --bboxes --include-ids
```

#### Detect Structure
```bash
uv run -m transform.detect_structure document.html blocks.json output_dir/
```

#### Clean HTML
```bash
uv run -m transform.clean_html -i main:content.html -o cleaned.html -k YOUR_API_KEY
```

## Project Structure

```
ccdr-ingestion-workflow/
├── main.py                    # Main pipeline orchestrator
├── transform/                 # Core transformation modules
│   ├── extract_text_blocks.py # Text extraction with styling
│   ├── extract_images.py      # Image extraction and description
│   ├── extract_svgs.py        # SVG extraction and description
│   ├── combine_blocks.py      # Block combination utilities
│   ├── convert_to_html.py     # Block-to-HTML conversion
│   ├── detect_structure.py    # Document structure detection
│   ├── clean_html.py          # LLM-based HTML cleaning
│   ├── html_to_graph.py       # HTML-to-graph conversion (WIP)
│   ├── create_relations.py    # Relationship extraction (WIP)
│   └── models.py              # Pydantic data models
├── sample_data/               # Sample data for different pipeline stages
├── schema_legacy.md           # Previous schema design
├── schema_revision.md         # Current schema design
└── pyproject.toml            # Project dependencies
```

## Key Features

### Multi-modal Content Extraction
- **Text**: Preserves styling and semantic structure from PDF
- **Images**: Automatic extraction with AI-generated descriptions
- **Vector Graphics**: SVG extraction with content analysis

### Intelligent Structure Detection
- Uses Gemini LLM to identify document sections
- Separates front matter, body, and back matter
- Maintains reading order and hierarchical relationships

### Semantic HTML Generation
- Limited tag vocabulary for consistent structure
- Rich data attributes for metadata preservation
- Support for academic document features (citations, footnotes, etc.)

### Concurrent Processing
- Parallel LLM API calls for image/SVG description
- Configurable concurrency limits to respect API rate limits
- Async/await patterns for efficient processing

### Flexible Output Formats
- JSON blocks for programmatic processing
- Structured HTML for human review
- Graph schema for database ingestion

## Dependencies

- **PyMuPDF**: PDF parsing and content extraction
- **Pillow**: Image processing
- **LiteLLM**: Unified LLM API interface
- **Pydantic**: Data validation and serialization
- **Tenacity**: Retry logic for API calls

## API Usage

The project uses two LLM providers:
- **Gemini**: Structure detection and image description
- **DeepSeek**: HTML cleaning and SVG description

Both APIs support concurrent requests with configurable rate limiting.

## Development Status

### Completed
- ✅ Text, image, and SVG extraction
- ✅ Block combination and HTML conversion
- ✅ Structure detection with LLM
- ✅ HTML cleaning and semantic conformance
- ✅ Concurrent processing with rate limiting

### In Progress
- 🚧 HTML-to-graph conversion (`html_to_graph.py`)
- 🚧 Relationship extraction from anchor tags (`create_relations.py`)
- 🚧 Database ingestion and storage

### Planned
- 📋 Evals for model selection (for SVG description and HTML cleaning)
- 📋 Logical page number mapping
- 📋 Cosine similarity for block-to-PDF mapping
- 📋 Vector embedding generation
- 📋 Full database schema implementation

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
