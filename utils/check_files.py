import re
import json
from pathlib import Path

def check_existing_files(pdf_path):
    """Check if markdown and related files already exist for a PDF"""
    base_name = Path(pdf_path).stem
    
    # Check for markdown file
    md_file = Path("data/parsed") / f"{base_name}.md"
    
    # Check for JSONL file (used by extraction)
    jsonl_file = Path("data/sections_report") / f"{base_name}.jsonl"
    
    # Check for embeddings
    embeddings_file = Path("data/embeddings") / f"{base_name}.faiss"
    
    return {
        'markdown': md_file.exists(),
        'jsonl': jsonl_file.exists(),
        'embeddings': embeddings_file.exists(),
        'md_path': md_file,
        'jsonl_path': jsonl_file,
        'embeddings_path': embeddings_file
    }

def determine_report_years(pdf1_path, pdf2_path):
    """Determine which PDF is 2023 and which is 2024 based on filename or content"""
    pdf1_name = Path(pdf1_path).stem.lower()
    pdf2_name = Path(pdf2_path).stem.lower()
    
    if "2024" in pdf1_name and "2023" in pdf2_name:
        return pdf1_path, pdf2_path 
    elif "2023" in pdf1_name and "2024" in pdf2_name:
        return pdf2_path, pdf1_path  
    elif "2024" in pdf1_name:
        return pdf1_path, pdf2_path 
    elif "2024" in pdf2_name:
        return pdf2_path, pdf1_path  
    else:
        print("Make sure 2024 pdf first and 2023 pdf second in arguments.")
        return pdf1_path, pdf2_path