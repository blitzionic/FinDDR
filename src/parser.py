from pydoc import doc
from pathlib import Path

import pandas as pd
import json
import os
import sys
import argparse

class DocProcessor:

    def __init__(self):
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions, TableStructureOptions, TableFormerMode, EasyOcrOptions
        )
        from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
        from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice

        pipe = PdfPipelineOptions()
        # OCR (toggle if you want auto OCR for scanned bits)
        pipe.do_ocr = True
        pipe.ocr_options = EasyOcrOptions(lang=["en"])  # add more langs if needed

        # Table structure (TablFormer)
        pipe.do_table_structure = True
        pipe.table_structure_options = TableStructureOptions(
            mode=TableFormerMode.ACCURATE,
            do_cell_matching=True
        )
        
        pdf_option = PdfFormatOption(
            # backend=DoclingParseV4DocumentBackend,  # or omit to use default
            pipeline_options=pipe
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: pdf_option
            }
        )

    def parse_once(self, pdf_path):
        result = self.converter.convert(pdf_path).document
        return result

    def process_to_markdown(self, parsed_doc):
        return parsed_doc.export_to_markdown()

    def process_to_tables(self, parsed_doc):
        tables_markdown = []
        for table_ix, table in enumerate(parsed_doc.tables, start=0):
            table_df: pd.DataFrame = table.export_to_dataframe()
            md = f"## Table {table_ix}\n\n{table_df.to_markdown(index=False)}\n\n"
            tables_markdown.append(md)
            
        return "".join(tables_markdown)

if __name__ == "__main__":
    # python parser.py data/nvidia_form_10-k.pdf
    parser = argparse.ArgumentParser(
        description='Process PDF files and extract markdown content using Docling'
    )

    parser.add_argument(
        'pdf_file', nargs='?', default='data/nvidia_form_10-k.pdf',
        help='Path to the PDF file to process (default: data/nvidia_form_10-k.pdf)'
    )
    parser.add_argument(
        '-o', '--output', default=None,
        help='Output markdown file name (default: auto-generated from input filename)'
    )
    parser.add_argument(
        '--no-print', action='store_true',
        help='Don\'t print markdown to console (only save to file)'
    )
    parser.add_argument(
        '--skip', action='store_true',
        help='Skip processing and only load markdown file'
    )
    
    args = parser.parse_args()

    if not args.skip:

        processor = DocProcessor()
        
        # Use the provided PDF file path
        pdf_path = args.pdf_file
        
        if os.path.exists(pdf_path):
            print(f"Processing {pdf_path}...")

            parsed = processor.parse_once(pdf_path)
            # get complete doc markdown
            output_md = parsed.export_to_markdown(parsed)
            # get table markdown
            # output_tables = processor.process_to_tables(parsed)

            # make output directory
            output_dir = Path("data/parsed")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if args.output:
                output_file = args.output
                output_dir = Path(output_file).parent
            else:
                output_dir = Path("data/parsed")
                base_name = Path(pdf_path).stem
                output_file = output_dir / f"{base_name}_parsed.md"
            
            output_dir.mkdir(parents=True, exist_ok=True)

            # Get document filename for exports
            doc_filename = Path(pdf_path).stem
            
            # Export Markdown format:
            md_file = output_dir / f"{doc_filename}_raw_parsed.md"
            with md_file.open("w", encoding="utf-8") as fp:
                fp.write(output_md)
            print(f"Markdown saved to: {md_file}")

            """
            tables_md = output_dir / f"{doc_filename}_table_parsed.md"
            with tables_md.open("w", encoding="utf-8") as fp:
                fp.write(output_tables)
            print(f"Tables saved to: {tables_md}")
            """


        else:
            print(f"PDF file not found: {pdf_path}")
            print("Please make sure the file exists and the path is correct.")

    # break into sections
    print(args.pdf_file)
    md_file = "data/parsed/" + args.pdf_file[5:-4] + "_raw_parsed.md"
    with open(md_file, "r", encoding="utf-8") as f:
        markdown_text = f.read()
        
    print(markdown_text[:500]) 
            
    print('\n')
    sections = extract_sections(markdown_text)
    print(f"Extracted {len(sections)} sections from markdown.")
