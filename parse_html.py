from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title
from unstructured.staging.base import elements_from_dicts
from typing import List, Dict
import os

def parse_html_file(file_path: str) -> List[Dict]:
    """Parses an HTML file using unstructured and returns a list of elements."""
    print(f"Parsing file: {file_path}...")
    try:
        # The core function from the 'unstructured' library
        elements = partition_html(filename=file_path, infer_table_structure=True, strategy='fast')
        # Convert the proprietary 'Element' objects to simple dictionaries
        return [el.to_dict() for el in elements]
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        return []
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []

if __name__ == "__main__":
    # Define the path to the file we downloaded in the previous step
    submission_file = "data/alphabet_10k_submission.txt"

    if not os.path.exists(submission_file):
        print(f"FATAL: The required submission file was not found: '{submission_file}'")
        print("Please run the `download_data.py` script first.")
    else:
        # Parse the file to get a list of its structural elements
        parsed_elements = parse_html_file(submission_file)

        if parsed_elements:
            print(f"\nSuccessfully parsed into {len(parsed_elements)} elements.")
            print("\n--- Sample Elements ---")

            # Print a few sample elements to inspect their type and content, just like in the article
            for i, element in enumerate(parsed_elements[20:25]):
                elem_type = element.get('type', 'N/A')
                # Get a snippet of the text content for preview
                text_snippet = element.get('text', '')[:100].replace('\n', ' ') + '...'
                print(f"Element {i+20}: [Type: {elem_type}] - Content: '{text_snippet}'")

            # --- STRUCTURE-AWARE CHUNKING ---
            
            # Step 2: Convert dictionary elements back to unstructured Element objects
            # This is necessary because the chunking functions expect this specific object type.
            elements_for_chunking = elements_from_dicts(parsed_elements)
            
            chunks = chunk_by_title(
                elements_for_chunking,
                max_characters=2048,
                combine_text_under_n_chars=256,
                new_after_n_chars=1800
            )
            
            print(f"\nDocument chunked into {len(chunks)} sections.")
            print("\n--- Sample Chunks ---")

            text_chunk_sample = None
            table_chunk_sample = None

            for chunk in chunks:
                if 'text_as_html' in chunk.metadata.to_dict() and table_chunk_sample is None:
                    table_chunk_sample = chunk
                elif 'text_as_html' not in chunk.metadata.to_dict() and text_chunk_sample is None and len(chunk.text) > 500:
                    text_chunk_sample = chunk
                
                if text_chunk_sample and table_chunk_sample:
                    break

            if text_chunk_sample:
                print("\n** Sample Text Chunk **")
                print(f"Content: {text_chunk_sample.text[:500]}...")
                print(f"Metadata: {text_chunk_sample.metadata.to_dict()}")

            if table_chunk_sample:
                print("\n** Sample Table Chunk **")
                html_content = table_chunk_sample.metadata.to_dict().get('text_as_html', '')
                print(f"HTML Content: {html_content[:500]}...")
                print(f"Metadata: {table_chunk_sample.metadata.to_dict()}")

            
