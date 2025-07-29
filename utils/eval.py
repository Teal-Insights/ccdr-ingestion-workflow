import json
import re
from statistics import mean
from transform.detect_nested_structure import ParsedHTMLPartial, Context

FILEPATH = "artifacts/validated_responses_20250729_171818.jsonl"

with open(FILEPATH, "r") as f:
    data = []
    for line in f:
        data.append(json.loads(line))

# {
#     "timestamp": datetime.now().isoformat(),
#     "model": actual_model,
#     "cost": float(cost),
#     "depth": depth,
#     "context": context.model_dump(),
#     "input_blocks_count": len(blocks),
#     "messages": messages,
#     "raw_response": response.choices[0].message.content,
#     "parsed_response": parsed_html_partial.model_dump(),
#     "validation_attempt": attempt + 1
# }

score_keys = ['share_disjoint', 'share_present', 'share_children_unique', 'valid_list_parents', 'valid_table_parents', 'valid_tr_parents', 'no_universal_wrapper_node', 'share_valid_img_sources']
scores = []

for item in data:
    item_scores = {}
    
    # Get all ids from the user message
    user_input = item["messages"][0]["content"].split("Content:")[1].strip()
    matches = re.findall(r'<(p|img) id="(\d+)"', user_input)
    tags = [match[0] for match in matches]
    ids = set([int(match[1]) for match in matches])

    # Get all children and sources from the parsed response
    children = []
    sources = []
    parsed_response = ParsedHTMLPartial.model_validate(item["parsed_response"])
    for node in parsed_response.proposed_nodes:
        children.extend(node.children)
        sources.extend(node.sources)

    # Get the context from the item
    parsed_context = Context.model_validate(item["context"])

    # Store model
    item_scores["model"] = item["model"]

    # Store user input length for weighting
    item_scores["user_input_length"] = len(user_input)

    # Are the children and sources lists non-disjoint?
    item_scores["share_disjoint"] = len(set(children).difference(set(sources)))/len(ids)

    # Are any ids not present in either the children or sources lists?
    item_scores["share_present"] = len(set(ids).intersection(set(children).union(set(sources))))/len(ids)

    # Are any children repeated more than once?
    item_scores["share_children_unique"] = len(set(children))/len(children) if children else None

    # Do any `li` tags not have `ul` or `ol` parents?
    if any(node.tag == "li" for node in parsed_response.proposed_nodes):
        item_scores["valid_list_parents"] = 0 if not any(tag in ["ul", "ol"] for tag in parsed_context.parent_tags) else 1
    else:
        item_scores["valid_list_parents"] = None

    # Do any `tr` tags not have `table` parents?
    if any(node.tag == "tr" for node in parsed_response.proposed_nodes):
        item_scores["valid_table_parents"] = 0 if not any(tag == "table" for tag in parsed_context.parent_tags) else 1
    else:
        item_scores["valid_table_parents"] = None

    # Do any `th` or `td` tags not have `tr` and `table` parents?
    if any(node.tag in ["th", "td"] for node in parsed_response.proposed_nodes):
        item_scores["valid_tr_parents"] = 0 if not any(tag == "tr" for tag in parsed_context.parent_tags) or not any(tag == "table" for tag in parsed_context.parent_tags) else 1
    else:
        item_scores["valid_tr_parents"] = None

    # Has the LLM proposed a single wrapper node that has all the indices as its children?
    if any(set(node.children) == ids for node in parsed_response.proposed_nodes):
        item_scores["no_universal_wrapper_node"] = 0
    else:
        item_scores["no_universal_wrapper_node"] = 1

    # Does any proposed `img` tag have an invalid source (valid meaning exactly one `img` tag)?
    if any(node.tag == "img" for node in parsed_response.proposed_nodes):
        original_img_ids = [id for tag, id in zip(tags, ids) if tag == "img"]
        num_proposed_images = len([node for node in parsed_response.proposed_nodes if node.tag == "img"])
        proposed_img_sources = [node.sources for node in parsed_response.proposed_nodes if node.tag == "img"]
        invalid_sources = [source for source in proposed_img_sources if len(source) != num_proposed_images or source not in original_img_ids]
        item_scores["share_valid_img_sources"] = 1 - len(invalid_sources)/num_proposed_images
    else:
        item_scores["share_valid_img_sources"] = None

    item_scores["average_score"] = mean(score for key, score in item_scores.items() if score is not None and key in score_keys)
    scores.append(item_scores)

max_length = max(score["user_input_length"] for score in scores)
for score in scores:
    score["weight"] = score["user_input_length"]/max_length

for score in scores:
    score["weighted_score"] = score["average_score"] * score["weight"]

# Save top 5 scoring objects to a JSON file
high_scorers_indices = sorted(range(len(scores)), key=lambda i: scores[i]["weighted_score"], reverse=True)[:5]
high_scorers = [data[i] for i in high_scorers_indices]
with open("artifacts/high_scorers.json", "w") as f:
    json.dump(high_scorers, f, indent=4)

# Print mean scores by model
for model in set(score["model"] for score in scores):
    model_scores = [score for score in scores if score["model"] == model]
    num_examples = len(model_scores)
    print(f"{model}: {mean(score['average_score'] for score in model_scores)} ({num_examples} examples)")