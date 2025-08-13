import os
import logging
import subprocess
from pathlib import Path
import dotenv
import shutil

from utils.models import ContentBlock, StructuredNode, TagName
from utils.html import create_nodes_from_html, validate_data_sources

# Load environment variables
dotenv.load_dotenv()

logger = logging.getLogger(__name__)

ALLOWED_TAGS: str = ", ".join(
    tag.value for tag in TagName
) + ", b, i, u, s, sup, sub, br"

HTML_PROMPT = f"""You are an expert in HTML and document structure.
You are given an HTML partial with a flat structure and only `p` and `img` tags.
Your task is to propose a better structured HTML representation of the content, with semantic tags and logical hierarchy.
Only `header`, `main`, and/or `footer` are allowed at the top level; all other tags should be nested within one of these.

# Requirements

- You may only use the following tags: {ALLOWED_TAGS}.
    - `header`: Front matter (title pages, table of contents, preface, etc.)
    - `main`: Body content (main chapters, sections, core content)
    - `footer`: Back matter (appendices, bibliography, index, etc.)
- You may split, merge, or replace structural containers as necessary, but you should make an effort to:
    - Clean up any whitespace, encoding, redundant style tags, or other formatting issues
    - Otherwise maintain the identical wording/spelling of the text content and of image descriptions and source URLs
    - Assign elements a `data-sources` attribute with a comma-separated list of ids of the source elements (for attribute mapping and content validation). This can include ranges, e.g., `data-sources="0-5,7,9-12"`
        - Note: inline style tags `b`, `i`, `u`, `s`, `sup`, `sub`, and `br` do not need `data-sources`, but all other tags should have this attribute

# Example

Here is an information-dense example illustrating some key transformations:

1.  **Nesting and Logical Headings:** Creating a hierarchical structure with `<section>`, `<h2>`, and `<h3>` tags based on the input's numbering and context.
2.  **Aggregation:** Combining multiple separate `<p>` tags from the input into a single, semantically correct `<table>` element in the output. The container `<table>` gets an aggregated `data-sources` attribute.
3.  **Splitting:** Deconstructing a single, complex `<p>` tag (which contains multiple headings, lists, and tabular data mashed together) into multiple structured elements in the output (`<h3>`, `<ul>`, `<table>`). Each of the new elements correctly inherits the `data-sources` from the original `<p>` tag.
4.  **List and Table Creation:** Identifying implicit lists and tables within the input text and formatting them correctly in the output.
5.  **Cleaning:** Removing redundant tags, whitespace, and other formatting issues while maintaining the original text content.

## Example input

```html
<p id="141">The persistence of weak governance, compounded by the economic crisis, undermines the capacity of the water sector to respond to the impacts of climate change. The water sector faces significant governance challenges that impede effective resource management and delivery of sustainable water services. Despite adoption of integrated water resource management principles, progress is considerably slower than the regional average. Complex legal frameworks and regulations have resulted in a convoluted institutional structure, diminishing accountability and transparency. Responsibilities for water management are fragmented across ministries and administrative levels, leading to overlap of responsibilities and lack of accountability. Coordination among stakeholders is deficient, and there is a notable absence of reliable data for informed planning, strategy development, policy formulation, and investment prioritization. Prolongation</p>
<p id="142">of the crisis will exacerbate the sector's limitations by preventing new investments in basic service delivery and the storage capacity needed to build resilience to projected climate change in the long term. The renewable water resource diminished from more than 2,000 cubic meters per capita in the 1960s to less than 660 cubic meters per capita in 2020. The "Muddling through" scenario in the water sector will have serious consequences that are highlighted in Table 3 .</p>
<p id="143">Table 3</p>
<p id="144"><b>Characteristics</b></p>
<p id="145"><b>Impacts</b></p>
<p id="146">•  Limited infrastructure to store water and bridge the seasonal gap in the availability of water.</p>
<p id="147">•  The operations of most of the country's about 75 wastewater treatment continue to be hindered by the high costs of energy and limited sewerage networks connectivity.</p>
<p id="148">•  The institutional capacity for building resilience to climate change is limited at the level of the water utility and the ministry.</p>
<p id="149">•  Tariffs are too low to cover operation and maintenance costs for existing infrastructure, address the high share of non-revenue water, the brain drain of human capital and lack of technical capabilities.</p>
<p id="150">•  The continuous brain drains of the human capital within the Water Establishments.</p>
<p id="151">•  More than 70 percent of the population face critical water shortages and an increasing cost of supply.</p>
<p id="152">•  Increases in the cost of water for the extremely income vulnerable groups.</p>
<p id="153">•  Poor quality of services undermines the willingness of the population to pay, while the increase in the cost of basic goods and services means that many are unable to pay.</p>
<p id="154">• Increased prevalence of, and inability to control, water borne diseases, as seen with the outbreak of cholera in October 2022</p>
<p id="195">3.5  Recommendations and Investment Needs</p>
<p id="196">Amid a deepening economic crisis, Lebanon has almost no fiscal space, limited institutional capacity, and numerous development challenges; therefore, it is critical to prioritize and sequence recommended measures  and  interventions—reflecting  their  urgency,  synergies,  and  trade-offs— in responding to development and climate needs. The recommended measures of this CCDR, responding to the above- <b>HIGH</b>lighted development and climate needs, are shown in Figure 12 . The recommended measures of this CCDR, responding to the above-<b>HIGH</b>lighted development and climate needs, are shown in Figure 12 .</p>
<p id="197">Figure 12: Country Climate and Development Report Recommended Policies and Interventions</p>
<p id="198"><b><b>HIGH</b> IMPACT AND RECOVERY-DRIVING INVESTMENTS</b> Urgent interventions with <b>LOW</b> tradeoffs Expand capacity of cleaner and affordable renewable energy sources and switch from liquid fuel to natural gas Increase water sector resilience by rehabilitating/expand water supply network, reducing non-revenue water, water supply reservoir, treatment plants, sewerage networks, rehabilitation of irrigation canals Promote solid waste sector sustainability by rehabilitate existing treatment facilities (sorting and composting plants), building new ones in case they do not exist, and investing in building and equipping sanitary landfills Enhance access and sustainability of transport by focusing on public transport service delivery and electrification, and ensuring climate resilience of roads and ports <b>Cost (US$ million)</b> Energy 4000 Water 1800 Solid waste 200 Transport 1580 Long-term interventions to keep in mind Less urgent, but <b>HIGH</b>ly beneﬁcial interventions Adopt climate policies to further reduce emissions intensity on track for net zero trajectory, as the tradeoff between development and climate would arise by mid 2030s Adoption of a nation-wide circular economy approach Implement Landfill Gas recovery systems to further reduce GHG emissions in designated SW sites Scope innovative technologies to improve wastewater treatment while promoting energy efficiency Promote climate action in health and education services and foster innovation Implement the Integrated Hydrological Information System Meteorological and hydrometric network expansion and improvement <b>LONG TERM MEASURES (BEYOND 2030)</b> Promote climate-smart and sustainable ecoturism Establish a funding mechanism with a dedicated revenue stream to support road network maintenance works Plan for Just Transition in collaboration with non-government stakeholders Adopt green procurement principies Formalize and consolidate public transport providers Adoption of a national composting strategy to reduce the amount of organic waste sent to landfills Develop a sustainable financing model for disaster monitoring and management <b>MEDIUM-TERM MEASURES</b> DE VEL OPMENT URGENC Y <b>HIGH</b> <b>LOW</b></p>
```

## Example output (probably nested in a `<main>` tag)

```html
  <section data-sources="141-154">
    <p data-sources="141-142">The persistence of weak governance, compounded by the economic crisis, undermines the capacity of the water sector to respond to the impacts of climate change. The water sector faces significant governance challenges that impede effective resource management and delivery of sustainable water services. Despite adoption of integrated water resource management principles, progress is considerably slower than the regional average. Complex legal frameworks and regulations have resulted in a convoluted institutional structure, diminishing accountability and transparency. Responsibilities for water management are fragmented across ministries and administrative levels, leading to overlap of responsibilities and lack of accountability. Coordination among stakeholders is deficient, and there is a notable absence of reliable data for informed planning, strategy development, policy formulation, and investment prioritization. Prolongation of the crisis will exacerbate the sector's limitations by preventing new investments in basic service delivery and the storage capacity needed to build resilience to projected climate change in the long term. The renewable water resource diminished from more than 2,000 cubic meters per capita in the 1960s to less than 660 cubic meters per capita in 2020. The "Muddling through" scenario in the water sector will have serious consequences that are highlighted in Table 3.</p>
    <table data-sources="143-154">
      <caption data-sources="143">Table 3</caption>
      <thead data-sources="144-145">
        <tr data-sources="144-145">
          <th data-sources="144">Characteristics</th>
          <th data-sources="145">Impacts</th>
        </tr>
      </thead>
      <tbody data-sources="146-154">
        <tr data-sources="146,151">
          <td data-sources="146">• Limited infrastructure to store water and bridge the seasonal gap in the availability of water.</td>
          <td data-sources="151">• More than 70 percent of the population face critical water shortages and an increasing cost of supply.</td>
        </tr>
        <tr data-sources="147,152">
          <td data-sources="147">• The operations of most of the country's about 75 wastewater treatment continue to be hindered by the high costs of energy and limited sewerage networks connectivity.</td>
          <td data-sources="152">• Increases in the cost of water for the extremely income vulnerable groups.</td>
        </tr>
        <tr data-sources="148,153">
          <td data-sources="148">• The institutional capacity for building resilience to climate change is limited at the level of the water utility and the ministry.</td>
          <td data-sources="153">• Poor quality of services undermines the willingness of the population to pay, while the increase in the cost of basic goods and services means that many are unable to pay.</td>
        </tr>
        <tr data-sources="149,154">
          <td data-sources="149">• Tariffs are too low to cover operation and maintenance costs for existing infrastructure, address the high share of non-revenue water, the brain drain of human capital and lack of technical capabilities.</td>
          <td data-sources="154">• Increased prevalence of, and inability to control, water borne diseases, as seen with the outbreak of cholera in October 2022</td>
        </tr>
        <tr data-sources="150">
          <td data-sources="150">• The continuous brain drains of the human capital within the Water Establishments.</td>
          <td data-sources="150"></td>
        </tr>
      </tbody>
    </table>
  </section>
  <section data-sources="195-198">
    <h2 data-sources="195">3.5 Recommendations and Investment Needs</h2>
    <p data-sources="196">Amid a deepening economic crisis, Lebanon has almost no fiscal space, limited institutional capacity, and numerous development challenges; therefore, it is critical to prioritize and sequence recommended measures and interventions—reflecting their urgency, synergies, and trade-offs— in responding to development and climate needs. The recommended measures of this CCDR, responding to the above-highlighted development and climate needs, are shown in Figure 12.</p>
    <figure data-sources="197-198">
      <figcaption data-sources="197">Figure 12: Country Climate and Development Report Recommended Policies and Interventions</figcaption>
      <h3 data-sources="198">High Impact and Recovery-Driving Investments</h3>
      <p data-sources="198">Urgent interventions with low tradeoffs:</p>
      <ul data-sources="198">
        <li data-sources="198">Expand capacity of cleaner and affordable renewable energy sources and switch from liquid fuel to natural gas.</li>
        <li data-sources="198">Increase water sector resilience by rehabilitating/expanding the water supply network, reducing non-revenue water, water supply reservoirs, treatment plants, sewerage networks, and rehabilitating irrigation canals.</li>
        <li data-sources="198">Promote solid waste sector sustainability by rehabilitating existing treatment facilities (sorting and composting plants), building new ones if they do not exist, and investing in building and equipping sanitary landfills.</li>
        <li data-sources="198">Enhance access and sustainability of transport by focusing on public transport service delivery and electrification, and ensuring climate resilience of roads and ports.</li>
      </ul>
      <table data-sources="198">
        <caption data-sources="198">Cost (US$ million)</caption>
        <tbody data-sources="198">
          <tr data-sources="198"><th data-sources="198">Energy</th><td data-sources="198">4000</td></tr>
          <tr data-sources="198"><th data-sources="198">Water</th><td data-sources="198">1800</td></tr>
          <tr data-sources="198"><th data-sources="198">Solid waste</th><td data-sources="198">200</td></tr>
          <tr data-sources="198"><th data-sources="198">Transport</th><td data-sources="198">1580</td></tr>
        </tbody>
      </table>
      <h3 data-sources="198">Medium-Term Measures</h3>
      <p data-sources="198">Less urgent, but highly beneficial interventions:</p>
      <ul data-sources="198">
        <li data-sources="198">Promote climate-smart and sustainable ecotourism.</li>
        <li data-sources="198">Establish a funding mechanism with a dedicated revenue stream to support road network maintenance works.</li>
        <li data-sources="198">Plan for a Just Transition in collaboration with non-government stakeholders.</li>
        <li data-sources="198">Adopt green procurement principles.</li>
        <li data-sources="198">Formalize and consolidate public transport providers.</li>
        <li data-sources="198">Adopt a national composting strategy to reduce the amount of organic waste sent to landfills.</li>
        <li data-sources="198">Develop a sustainable financing model for disaster monitoring and management.</li>
      </ul>
      <h3 data-sources="198">Long-Term Measures (Beyond 2030)</h3>
      <ul data-sources="198">
        <li data-sources="198">Adopt climate policies to further reduce emissions intensity on a track for a net-zero trajectory, as the tradeoff between development and climate would arise by the mid-2030s.</li>
        <li data-sources="198">Adopt a nation-wide circular economy approach.</li>
        <li data-sources="198">Implement Landfill Gas recovery systems to further reduce GHG emissions in designated SW sites.</li>
        <li data-sources="198">Scope innovative technologies to improve wastewater treatment while promoting energy efficiency.</li>
        <li data-sources="198">Promote climate action in health and education services and foster innovation.</li>
        <li data-sources="198">Implement the Integrated Hydrological Information System.</li>
        <li data-sources="198">Expand and improve the meteorological and hydrometric network.</li>
      </ul>
    </figure>
  </section>
```

# File Output Instructions

You MUST restructure the input HTML and save the complete restructured result to the specified output file path.

Please read the input HTML file carefully, analyze its structure, and create a well-structured HTML output that follows all the requirements above. Save the complete restructured HTML to the output file.

Remember:
- Every element (except inline style tags b, i, u, s, sup, sub, br) MUST have a data-sources attribute
- The data-sources values must comprehensively cover all IDs from the input HTML
- Use semantic HTML structure with proper nesting
- Clean up formatting issues while preserving exact text content
"""


def restructure_with_claude_code(
    content_blocks: list[ContentBlock],
    output_file: str = "output.html",
    working_dir: Path = Path("wkdir")
) -> list[StructuredNode]:
    """
    Restructure HTML using Claude Code in non-interactive mode.
    
    This approach is more token-efficient by:
    1. Writing input HTML to a file rather than including it in prompts
    2. Using Claude Code with limited tool permissions 
    3. Running validation via hooks automatically
    4. Reading the structured output from a file
    
    Args:
        content_blocks: List of content blocks to restructure
        output_file: Name of output file (will be placed in wkdir/)
        
    Returns:
        List of structured nodes parsed from output
    """
    # Ensure working directory exists
    working_dir.mkdir(exist_ok=True)

    # Copy contents of claude_config directory to working directory
    shutil.copytree(Path(__file__).parent.parent / "claude_config", working_dir, dirs_exist_ok=True)
    
    # Generate input HTML and write to file
    input_html = "\n".join([block.to_html(block_id=i) for i, block in enumerate(content_blocks)])
    input_file_path = working_dir / "input.html"
    output_file_path = working_dir / output_file
    
    # Write input HTML to file
    with open(input_file_path, "w", encoding="utf-8") as f:
        f.write(input_html)
    
    logger.info(f"Input HTML written to {input_file_path}")
    
    # Create the prompt that references the input file instead of including content
    file_prompt = f"""Please restructure the HTML content from the input file: {input_file_path.absolute()}

Save the complete restructured HTML to: {output_file_path.absolute()}

{HTML_PROMPT}"""
    
    # Run Claude Code in non-interactive mode with limited permissions
    try:
        # Use Claude Code with direct path (since it's an alias in shell)
        claude_path = str(Path.home() / ".claude/local/claude")
        cmd = [
            claude_path,
            "-p", file_prompt,
            "--allowedTools", "Edit", "MultiEdit", "Read",
            "--disallowedTools", "Bash", "Grep", "LS", "Glob",
            "--dangerously-skip-permissions"
        ]
        
        logger.info(f"Running Claude Code with command: {' '.join(cmd)}")
        
        # Prepare environment with API key and config directory
        env = {**os.environ}
        
        # Ensure API key is available - try ANTHROPIC_API_KEY first, then OPENROUTER_API_KEY
        assert "ANTHROPIC_API_KEY" in env, "ANTHROPIC_API_KEY not found in environment variables"
        
        result = subprocess.run(
            cmd,
            cwd=working_dir,  # Run from wkdir
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
            env=env
        )
        
        if result.returncode != 0:
            error_msg = f"Claude Code failed with return code {result.returncode}"
            if result.stderr:
                error_msg += f"\nStderr: {result.stderr}"
            if result.stdout:
                error_msg += f"\nStdout: {result.stdout}"
            raise RuntimeError(error_msg)
        
        logger.info("Claude Code completed successfully")
        logger.debug(f"Claude Code output: {result.stdout}")
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("Claude Code timed out after 5 minutes")
    except FileNotFoundError:
        raise RuntimeError("Claude Code not found. Please ensure 'claude' is installed and in PATH.")
    
    # Verify output file was created
    if not output_file_path.exists():
        raise RuntimeError(f"Output file was not created: {output_file_path}")

    # Read and parse the output
    with open(output_file_path, "r", encoding="utf-8") as f:
        restructured_html = f.read()

    # Validate data sources
    missing_ids, extra_ids = validate_data_sources(input_html, restructured_html)
    if (len(missing_ids) + len(extra_ids)) > 3:
        raise RuntimeError(f"Data sources validation failed: {missing_ids} {extra_ids}")

    # If it contains <body></body>, split on that and return the interior as the restructured HTML
    if "<body>" in restructured_html and "</body>" in restructured_html:
        restructured_html = restructured_html.split("<body>")[1].split("</body>")[0]

    # Parse into structured nodes
    nodes = create_nodes_from_html(restructured_html, content_blocks)
    
    logger.info(f"Successfully restructured HTML with {len(nodes)} top-level nodes")
    
    return nodes


if __name__ == "__main__":
    import json
    import dotenv
    import time
    
    dotenv.load_dotenv()
    
    # Load content blocks
    input_file = Path("artifacts") / "doc_601_content_blocks_with_styles.json"
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        exit(1)
    
    with open(input_file, "r") as fr:
        content_blocks_data = json.load(fr)
        content_blocks = [ContentBlock.model_validate(block) for block in content_blocks_data]
    
    logger.info(f"Loaded {len(content_blocks)} content blocks")
    
    # Run restructuring
    start_time = time.time()
    try:
        structured_nodes = restructure_with_claude_code(
            content_blocks,
            output_file="output.html",
            working_dir=Path("doc_601")
        )
        end_time = time.time()
        
        # Save structured nodes to JSON
        output_json = Path("artifacts") / "doc_601_nested_structure.json"
        with open(output_json, "w") as fw:
            json.dump([node.model_dump() for node in structured_nodes], fw, indent=2)
        
        logger.info(f"Process completed in {end_time - start_time:.2f} seconds")
        logger.info(f"HTML output saved to {Path('doc_601') / 'output.html'}")
        logger.info("JSON output saved to artifacts/doc_601_nested_structure.json")
        logger.info(f"Structured data saved to {output_json}")
        
    except Exception as e:
        logger.error(f"Restructuring failed: {e}")
        exit(1)
