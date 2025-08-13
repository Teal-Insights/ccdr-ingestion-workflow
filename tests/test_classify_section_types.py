import json
from utils.models import StructuredNode
from utils.schema import TagName

file_to_check = "artifacts/doc_601_nested_structure_classified.json"

with open(file_to_check, "r") as f:
    data = json.load(f)

nodes = [StructuredNode.model_validate(node) for node in data]

for node in nodes:
    if node.tag == TagName.SECTION:
        print(node.section_type)