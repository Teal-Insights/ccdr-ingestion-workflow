import asyncio
import json
import logging
import os
import re
import argparse
import time
from typing import Optional

import dotenv
from litellm import Router, acompletion
from pydantic import BaseModel, Field, ValidationError, field_validator

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
dotenv.load_dotenv()


# --- Pydantic Models for Structured I/O ---

class EvaluationScore(BaseModel):
    """A score and a justification for a single evaluation criterion."""
    score: int = Field(..., description="A score from 1 (poor) to 5 (excellent).", ge=1, le=5)
    reasoning: str = Field(..., description="A brief but specific justification for the score.")

class JudgeEvaluation(BaseModel):
    """Structured evaluation from the judge LLM."""
    structure_logic: EvaluationScore = Field(description="Assesses the logical nesting of tags (e.g., 'li' in 'ul', 'tr' in 'table').")
    coverage_and_disjointness: EvaluationScore = Field(description="Checks if all original IDs are used exactly once across 'children' and 'sources', and that these sets are disjoint.")
    rule_adherence: EvaluationScore = Field(description="Evaluates adherence to specific constraints like avoiding a single wrapper node and correct image tag sourcing.")
    content_preservation: EvaluationScore = Field(description="Judges how well the original text and inline tags (b, i, etc.) were preserved.")
    overall_quality: EvaluationScore = Field(description="A holistic assessment of the response's quality and usefulness as a few-shot example.")
    
    average_score: Optional[float] = None # We will calculate this ourselves

    @field_validator('average_score', mode='before')
    @classmethod
    def calculate_average_score(cls, v, values):
        """Calculate the average score from the other scores."""
        scores = [
            values.data['structure_logic'].score,
            values.data['coverage_and_disjointness'].score,
            values.data['rule_adherence'].score,
            values.data['content_preservation'].score,
            values.data['overall_quality'].score
        ]
        return sum(scores) / len(scores) if scores else 0.0

class ParsedProposedNode(BaseModel):
    """A simplified model to parse the 'parsed_response' from logs."""
    tag: str
    children: list[int]
    sources: list[int]
    text: Optional[str] = None

class LogEntry(BaseModel):
    """Represents a single line/entry in the input validated_responses.jsonl file."""
    model: str
    messages: list[dict[str, str]]
    parsed_response: dict[str, list[ParsedProposedNode]]


# --- LLM and Prompting Logic ---

JUDGE_PROMPT_TEMPLATE = """
You are an expert AI assistant specializing in evaluating the quality of other AI models' outputs based on a strict set of rules. Your analysis must be precise, objective, and your response must be in the specified JSON format.

## The Task

An AI model was given a list of HTML blocks (identified by `<!-- id: ... -->`) and was asked to restructure them into a more semantically correct hierarchy.

Your job is to evaluate how well the model performed this task based on its proposed JSON output.

## Original Input Given to the Model

```html
{html_representation}
```

## The Model's Proposed Structure (in JSON)

```json
{proposed_structure}
```

## Evaluation Criteria

Please evaluate the model's proposal against the following criteria. For each criterion, provide a score from 1 (poor) to 5 (excellent) and a brief justification for your score.

1.  **Structure Logic**: Is the proposed hierarchy logical?
    - Are list items (`li`) correctly placed within list containers (`ul`, `ol`)?
    - Are table components (`th`, `td`, `tr`) correctly placed within table containers (`thead`, `tbody`, `table`)?
    - Is the overall structure a clear improvement over a flat list of blocks?
    - Does the structure avoid overcomplicating the hierarchy with unnecessary layers of abstraction (e.g., wrapping figures in a table)?

2.  **Coverage and Disjointness**: Are the `children` and `sources` lists managed correctly?
    - **Coverage**: Do the `sources` and `children` lists, when combined across all proposed nodes, account for *every* original block ID exactly once? (The number of original blocks is {block_count}).
    - **Disjointness**: For any single proposed node, is there any overlap between its `children` and `sources`? (There should be none). Are the `sources` and `children` sets across the *entire* proposal disjoint? (They must be).

3.  **Rule Adherence**: Did the model follow specific positive and negative constraints?
    - Does the proposal avoid creating a single, lazy wrapper node (e.g., one `div`) that contains all original blocks as its children?
    - Are `img` tags proposed correctly (e.g., having `sources` but no `children` and no `text`)?

4.  **Content Preservation**: Was the original text content and inline formatting handled correctly?
    - Is the text from the source blocks faithfully moved into the `text` field of new leaf nodes?
    - Are inline HTML tags like `<b>`, `<i>`, `<u>` preserved within the `text` field?
    - Is it free from major omissions or hallucinations?

5.  **Overall Quality**: A holistic judgment.
    - How confident are you that this is a high-quality transformation?
    - Would you recommend this response as a "gold standard" few-shot example for training other models?

## Response Format

You **MUST** return your response in the following JSON format. Do not add any text before or after the JSON object.

```json
{json_schema}
```
"""

def create_judge_router(deepseek_api_key: str) -> Router:
    """Creates a LiteLLM router configured to use a powerful model for judging."""
    model_list = [
        {
            "model_name": "judge-llm",
            "litellm_params": {
                # Use a powerful model for nuanced evaluation tasks
                "model": "deepseek/deepseek-reasoner", 
                "api_key": deepseek_api_key,
            }
        }
    ]
    return Router(model_list=model_list, num_retries=2)

def extract_html_from_log(log_entry: LogEntry) -> tuple[str, int]:
    """Extracts the original HTML representation from the logged messages."""
    try:
        user_prompt_content = log_entry.messages[0]['content']
        # Use regex to find the content within the ```html ... ``` block
        match = re.search(r'```html\s*\n(.*?)\n```', user_prompt_content, re.DOTALL)
        if not match:
            logger.warning("Could not find ```html block in logged prompt.")
            return "--- HTML NOT FOUND ---", 0
        
        html_content = match.group(1).strip()
        # Count the number of blocks by finding the highest ID
        ids = re.findall(r'<!-- id: (\d+) -->', html_content)
        block_count = max(int(i) for i in ids) + 1 if ids else 0
        return html_content, block_count
    except (IndexError, KeyError, AttributeError) as e:
        logger.error(f"Error parsing log entry for HTML content: {e}")
        return "--- HTML PARSING ERROR ---", 0

async def evaluate_response(
    log_entry: LogEntry, 
    router: Router
) -> Optional[JudgeEvaluation]:
    """Sends a single logged response to the judge LLM and returns the evaluation."""
    html_representation, block_count = extract_html_from_log(log_entry)
    if block_count == 0:
        return None

    proposed_structure_json = json.dumps(log_entry.parsed_response, indent=2)

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        html_representation=html_representation,
        proposed_structure=proposed_structure_json,
        block_count=block_count,
        json_schema=json.dumps(JudgeEvaluation.model_json_schema(), indent=2)
    )

    messages = [{"role": "user", "content": prompt}]

    try:
        response = await acompletion(
            model="judge-llm",
            messages=messages,
            response_format={"type": "json_object"},
            router=router,
            temperature=0.0
        )
        response_content = response.choices[0].message.content
        if not response_content:
            logger.error("LLM returned an empty response.")
            return None

        evaluation = JudgeEvaluation.model_validate_json(response_content)
        # Recalculate average score locally to be certain
        evaluation.average_score = evaluation.model_validate(evaluation.model_dump()).average_score
        return evaluation

    except ValidationError as e:
        logger.error(f"Validation Error from judge LLM response: {e}")
        logger.debug(f"Failed response content: \n{response_content}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during LLM call: {e}")
        return None

# --- Main Execution Logic ---

async def main(args):
    """Main function to read logs, evaluate them, and write results."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    judge_router = create_judge_router(openai_api_key)

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input_file}")
        return

    logger.info(f"Found {len(lines)} responses to evaluate in {args.input_file}.")
    
    # Use append mode to allow for resuming
    with open(args.output_file, 'a', encoding='utf-8') as f_out:
        for i, line in enumerate(lines):
            logger.info(f"--- Processing response {i+1}/{len(lines)} ---")
            if not line.strip():
                continue
            
            try:
                log_data = json.loads(line)
                log_entry = LogEntry.model_validate(log_data)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Skipping line {i+1} due to parsing error: {e}")
                continue

            evaluation_result = await evaluate_response(log_entry, judge_router)

            if evaluation_result:
                logger.info(f"Evaluation successful. Average Score: {evaluation_result.average_score:.2f}")
                
                # Combine original log with the new evaluation
                combined_output = {
                    "original_log": log_data,
                    "judge_evaluation": evaluation_result.model_dump()
                }
                
                f_out.write(json.dumps(combined_output, indent=None) + '\n')
                f_out.flush() # Ensure it's written immediately
            else:
                logger.warning(f"Failed to get a valid evaluation for line {i+1}.")
            
            # Simple rate limiting to be kind to APIs
            await asyncio.sleep(1) 

    logger.info(f"Evaluation complete. Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate LLM responses from a JSONL log file using a judge LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input-file", 
        type=str, 
        required=True,
        help="Path to the input .jsonl file containing validated responses."
    )
    parser.add_argument(
        "-o", "--output-file", 
        type=str, 
        default="evaluated_responses.jsonl",
        help="Path to the output .jsonl file to save evaluation results."
    )
    
    args = parser.parse_args()

    start_time = time.time()
    asyncio.run(main(args))
    end_time = time.time()
    logger.info(f"Total time taken: {end_time - start_time:.2f} seconds")
