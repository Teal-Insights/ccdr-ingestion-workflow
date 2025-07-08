# Convert HTML to graph, but leave text formatting and anchor tags as plain text
# Example implementation:
from lxml import html, etree
import json


class LXMLConverter:
    def __init__(self):
        self.nodes = []
        self.node_counter = 0

    def parse_html(self, html_content):
        tree = html.fromstring(html_content)
        self.traverse_element(tree)
        return self.nodes

    def traverse_element(self, element, parent_id=None, depth=0):
        node_id = self.node_counter
        self.node_counter += 1

        # Handle element
        node_data = {
            "id": node_id,
            "parent_id": parent_id,
            "node_type": "ELEMENT_NODE",
            "tag_name": element.tag.upper() if element.tag else None,
            "depth": depth,
            "attributes": dict(element.attrib),
        }

        # Handle text content
        if element.text and element.text.strip():
            node_data["text_content"] = element.text.strip()

        self.nodes.append(node_data)

        # Process children
        for i, child in enumerate(element):
            self.traverse_element(child, node_id, depth + 1)

            # Handle tail text (text after closing tag)
            if child.tail and child.tail.strip():
                tail_node_id = self.node_counter
                self.node_counter += 1
                self.nodes.append(
                    {
                        "id": tail_node_id,
                        "parent_id": node_id,
                        "node_type": "TEXT_NODE",
                        "text_content": child.tail.strip(),
                        "depth": depth + 1,
                        "sequence_in_parent": i + 1,
                    }
                )

        return node_id
