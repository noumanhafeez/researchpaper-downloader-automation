import os
import requests
import io
from urllib.parse import urlparse
from PyPDF2 import PdfReader, PdfWriter

def get_pdf_reader(input_source):
    base_filename = "output"
    response = requests.get(input_source, stream=True, timeout=30)
    response.raise_for_status()

    # Get filename from URL path
    parsed_url = urlparse(input_source)
    path_part = os.path.basename(parsed_url.path)
    if path_part and '.' in path_part:
         base_filename = os.path.splitext(path_part)[0]

    # Read content into memory
    pdf_content = io.BytesIO(response.content)
    reader = PdfReader(pdf_content)
    total_pages = len(reader.pages)
    return reader, base_filename, total_pages

def split_pdf(input_source, output_dir, pages_per_chunk):
    reader, base_filename, total_pages = get_pdf_reader(input_source)

    if reader is None:
        print("Failed to get PDF reader. Aborting split.")
        return

    try:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory '{output_dir}' ensured.")

        # Calculate the number of chunks
        num_chunks = (total_pages + pages_per_chunk - 1) // pages_per_chunk
        print(f"Splitting into {num_chunks} chunks of max {pages_per_chunk} pages each.")

        # Process each chunk
        for i in range(num_chunks):
            writer = PdfWriter()
            start_page = i * pages_per_chunk
            # Ensure end_page doesn't exceed total_pages
            end_page = min(start_page + pages_per_chunk, total_pages)

            print(f"Processing chunk {i+1}/{num_chunks} (pages {start_page + 1}-{end_page})...")

            # Add pages to the new PDF chunk
            for page_num in range(start_page, end_page):
                writer.add_page(reader.pages[page_num])

            # Construct the output filename
            output_filename = os.path.join(output_dir, f"{base_filename}_chunk_{i+1}.pdf")

            # Write the chunk to a new PDF file
            with open(output_filename, 'wb') as outfile:
                writer.write(outfile)
            print(f"Chunk {i+1} saved as '{output_filename}'")

        print("\nPDF splitting completed successfully!")

    except Exception as e:
        print(f"An error occurred during the splitting process: {e}")

split_pdf('https://store.veritaspress.com/site/assets/000775-secret-garden-comp-guide-lis.pdf',
output_dir='pdf_chunks', pages_per_chunk=2)