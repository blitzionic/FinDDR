import os
import re
import json

from pathlib import Path
from parser import DocProcessor
import argparse




def main():
    
    parser = DocProcessor()

    arg_parser = argparse.ArgumentParser(description="Process PDF files")
    arg_parser.add_argument('pdf_file1', nargs='?')
    arg_parser.add_argument('pdf_file2', nargs='?')
    args = arg_parser.parse_args()
    
    pdf1 = args.pdf_file1
    pdf2 = args.pdf_file2

    print(f"\n{'='*50}")
    print(f"Processing report for: {pdf1} and {pdf2}")
    print(f"{'='*50}")

    try:
        markdown_report1 = parser.process_document_to_markdown(pdf1)
        markdown_report2 = parser.process_document_to_markdown(pdf2)
        


        print(f"Successfully generated report!!")

    except Exception as e:
        print(f"❌ Error processing {pdf1} and {pdf2}: {str(e)}")
        exit(1)
        

if __name__ == "__main__":
    main()