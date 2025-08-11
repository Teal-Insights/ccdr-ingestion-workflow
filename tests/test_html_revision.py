import glob
import bs4
from utils.range_parser import parse_range_string


def test_that_html_revision_coverage_is_comprehensive():
    input_files = sorted(glob.glob("artifacts/revisions/input_*.html"))
    output_files = sorted(glob.glob("artifacts/revisions/output_*.html"))
    assert len(input_files) == len(output_files), "Input and output files have different numbers of files"
    
    # Drop pairs for any output files with "gpt-5" in the name
    pairs = zip(input_files, output_files)
    pairs = [pair for pair in pairs if "gpt-5" not in pair[1]]
    
    print(f"Found {len(pairs)} input and output files")
    
    # Check that the input and output HTML files have the same ids for each element
    files_with_mistmatched_ids: list[tuple[str, str]] = []
    for input_file, output_file in pairs:
        with open(input_file, "r") as f:
            input_html = f.read()
        with open(output_file, "r") as f:
            output_html = f.read()

        input_soup = bs4.BeautifulSoup(input_html, "html.parser")
        output_soup = bs4.BeautifulSoup(output_html, "html.parser")

        ids_in_input: set[int] = set(int(element.attrs["id"]) for element in input_soup.find_all() if "id" in element.attrs)
        ids_in_output: set[int] = set()

        # Get every HTML element's data-sources attribute and parse it into a list of integers, then update the set, and finally check that it is equal to the set of ids in the input HTML
        for element in output_soup.find_all():
            if "data-sources" in element.attrs:
                ids_in_output.update(parse_range_string(element["data-sources"]))

        if ids_in_input != ids_in_output:
            files_with_mistmatched_ids.append((input_file, output_file))
            print(f"ids in {input_file} not in {output_file}: {ids_in_input - ids_in_output}")
            print(f"ids in {output_file} not in {input_file}: {ids_in_output - ids_in_input}")

    assert len(files_with_mistmatched_ids) == 0, f"{len(files_with_mistmatched_ids)} pairs of files have different id numbers: {", ".join(pair[0] + " and " + pair[1] for pair in files_with_mistmatched_ids)}"