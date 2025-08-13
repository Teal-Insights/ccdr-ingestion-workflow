from litellm import Router, ModelResponse, Choices
import pydantic

from utils.models import StructuredNode
from utils.schema import TagName, SectionType

class SectionTypeClassification(pydantic.BaseModel):
    reasoning: str
    section_type: SectionType


async def _classify_section_type(router: Router, text: str, context: str) -> SectionType | None:
    """Use LLM to classify the type of section"""

    prompt = f"""You will be given an HTML section extracted from a PDF document (one of the World Bank's Country Climate and Development Reports).
For context, you will also be given the parent tags of the section.

Your job is to classify the section into *one* of the following types (the one that most closely matches how the section functions in the document):
{", ".join([f"{type.value}" for type in SectionType])}

Section to classify: 

```html
{text}
```

Parent tags: "{context}"

Return JSON with:
- reasoning: brief explanation of your classification
- section_type: one of the types above"""

    message = [{
        "role": "user", 
        "content": prompt
    }]

    response = await router.acompletion(
        model="text-classifier",
        messages=message,  # type: ignore
        temperature=0.0,
        response_format=SectionTypeClassification
    )

    try:
        if (response and isinstance(response, ModelResponse) and 
            isinstance(response.choices[0], Choices) and 
            response.choices[0].message.content):
            classification = SectionTypeClassification.model_validate_json(response.choices[0].message.content)
            return classification.section_type
        else:
            raise ValueError("Invalid response structure from router")
    except Exception as e:
        print(f"Error processing text classification response: {e}")
        return None


async def classify_section_types(router: Router, nodes: list[StructuredNode], context: str) -> list[StructuredNode]:
    """Classify the type of section"""
    classified_nodes = []
    for node in nodes:
        if node.tag == TagName.SECTION:
            node.section_type = await _classify_section_type(router, node.to_html(), context)
        if node.children:
            node.children = await classify_section_types(router, node.children, context)

        classified_nodes.append(node)

    return classified_nodes


if __name__ == "__main__":
    import os
    import json
    from pathlib import Path
    import dotenv
    import asyncio
    from litellm import Router

    from utils.litellm_router import create_router

    dotenv.load_dotenv()
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    assert OPENROUTER_API_KEY, "OPENROUTER_API_KEY is not set"

    router: Router = create_router("", "", "", openrouter_api_key=OPENROUTER_API_KEY)

    # Load nested structure from JSON (for testing purposes)
    file_path = os.path.join("artifacts", "doc_601_nested_structure.json")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Parse into StructuredNode objects
    nested_structure = [StructuredNode.model_validate(item) for item in data]

    # Classify section types
    nested_structure = asyncio.run(classify_section_types(router, nested_structure, ""))

    # Save structured nodes to JSON
    output_json = Path("artifacts") / "doc_601_nested_structure_classified.json"
    with open(output_json, "w") as fw:
        json.dump([node.model_dump() for node in nested_structure], fw, indent=2)