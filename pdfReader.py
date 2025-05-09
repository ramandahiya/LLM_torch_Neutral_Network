from PyPDF2 import PdfReader
import os

#   - Iterate through your PDF files.
#   - Extract text content.
#   - Clean the text.


def extract_text_from_pdf(pdf_path):
    """Extracts text content from a single PDF file."""
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text
 
def dump_pdf_text_to_file(pdf_directory, output_file):
    """
    Iterates through PDF files in a directory, extracts their text,
    and writes it to a single output file.
    """
    all_text = ""
    pdf_count = 0
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            text = extract_text_from_pdf(pdf_path)
            if text:  # Only add text if extraction was successful
                all_text += text + "\n\n"  # Add some separation between files
                pdf_count += 1
            else:
                print(f"No text extracted from: {filename}")
 
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(all_text)
 
    print(f"Successfully extracted text from {pdf_count} PDF files and saved it to: {output_file}")
 



# ... (rest of the BasicLLM code) ...
PDF_DIR = "pdf"
ALL_DATA_PATH = "dump/file.txt"

if __name__ == "__main__":
    pdf_directory = PDF_DIR  # input("Enter the path to the directory containing your PDF files: ")
    output_filename = ALL_DATA_PATH # input("Enter the name for the output text file (e.g., combined_text.txt): ")
 
    if os.path.isdir(pdf_directory):
        dump_pdf_text_to_file(pdf_directory, output_filename)
    else:
        print(f"Error: Directory '{pdf_directory}' not found.")