import os
import sys
from pathlib import Path
import time
from typing import List, Tuple
import argparse
from extraction_rag import extract, Lang

def find_markdown_files(directory: str) -> List[Path]:
    """Find all markdown files in the specified directory."""
    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    md_files = list(directory_path.glob("*.md"))
    return sorted(md_files)

def group_company_reports(md_files: List[Path]) -> List[Tuple[Path, Path]]:
    """
    Group markdown files by company, assuming naming convention: company_year.md
    Returns list of tuples (2024_file, 2023_file)
    """
    company_groups = {}
    
    for file_path in md_files:
        filename = file_path.stem
        parts = filename.split('_')
        
        if len(parts) >= 2:
            company = '_'.join(parts[:-1])  # Everything except the last part (year)
            year = parts[-1]
            
            if company not in company_groups:
                company_groups[company] = {}
            
            company_groups[company][year] = file_path
    
    # Create pairs of (2024, 2023) files
    pairs = []
    for company, years in company_groups.items():
        if '2024' in years and '2023' in years:
            pairs.append((years['2024'], years['2023']))
        else:
            print(f"⚠️  Warning: Company '{company}' missing 2024 or 2023 report")
            if '2024' in years:
                print(f"   Found only: {years['2024']}")
            if '2023' in years:
                print(f"   Found only: {years['2023']}")
    
    return pairs

def run_extraction_batch(directory: str, currency_code: str = "USD", target_lang: Lang = Lang.EN):
    """
    Run extraction on all markdown file pairs in the specified directory.
    
    Args:
        directory: Path to directory containing markdown files
        currency_code: Currency code for financial data
        target_lang: Target language for extraction
    """
    print(f"🔍 Scanning directory: {directory}")
    
    try:
        md_files = find_markdown_files(directory)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    if not md_files:
        print(f"❌ No markdown files found in {directory}")
        return
    
    print(f"📄 Found {len(md_files)} markdown files")
    
    # Group files by company
    file_pairs = group_company_reports(md_files)
    
    if not file_pairs:
        print("❌ No valid company pairs (2024 + 2023) found")
        return
    
    print(f"🏢 Found {len(file_pairs)} company pairs to process")
    
    # Process each company pair
    start_time = time.time()
    successful_extractions = 0
    failed_extractions = 0
    
    for i, (file_2024, file_2023) in enumerate(file_pairs, 1):
        print(f"\n{'='*80}")
        print(f"🏢 PROCESSING COMPANY {i}/{len(file_pairs)}")
        print(f"📊 2024 Report: {file_2024.name}")
        print(f"📊 2023 Report: {file_2023.name}")
        print(f"🌐 Language: {target_lang.name}")
        print(f"💱 Currency: {currency_code}")
        print(f"{'='*80}")
        
        try:
            # Run extraction for this company
            extract(
                md_file1=str(file_2024),
                md_file2=str(file_2023),
                currency_code=currency_code,
                target_lang=target_lang
            )
            successful_extractions += 1
            print(f"✅ SUCCESS: Completed extraction for {file_2024.stem.split('_')[0]}")
            
        except Exception as e:
            failed_extractions += 1
            print(f"❌ FAILED: Error processing {file_2024.stem.split('_')[0]}")
            print(f"   Error: {str(e)}")
            # Continue with next company instead of stopping
            continue
    
    # Print final summary
    end_time = time.time()
    total_duration = end_time - start_time
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)
    
    print(f"\n{'='*80}")
    print("📊 BATCH EXTRACTION SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successful extractions: {successful_extractions}")
    print(f"❌ Failed extractions: {failed_extractions}")
    print(f"📊 Total companies processed: {successful_extractions + failed_extractions}")
    print(f"⏱️  Total execution time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    
    if successful_extractions > 0:
        avg_time = total_duration / successful_extractions
        avg_minutes = int(avg_time // 60)
        avg_seconds = int(avg_time % 60)
        print(f"⏱️  Average time per company: {avg_minutes:02d}:{avg_seconds:02d}")
    
    print(f"📁 Results saved in: artifacts/")

def main():
    parser = argparse.ArgumentParser(description="Batch extract data from markdown reports")
    parser.add_argument(
        "directory", 
        help="Directory containing markdown files"
    )
    parser.add_argument(
        "--currency", 
        default="USD", 
        help="Currency code for financial data (default: USD)"
    )
    parser.add_argument(
        "--lang", 
        choices=["EN", "ZH_SIM", "ZH_TR"], 
        default="EN",
        help="Target language for extraction (default: EN)"
    )
    
    args = parser.parse_args()
    
    # Convert string to Lang enum
    lang_map = {
        "EN": Lang.EN,
        "ZH_SIM": Lang.ZH_SIM,
        "ZH_TR": Lang.ZH_TR
    }
    target_lang = lang_map[args.lang]
    
    print(f"🚀 Starting batch extraction")
    print(f"📁 Directory: {args.directory}")
    print(f"💱 Currency: {args.currency}")
    print(f"🌐 Language: {args.lang}")
    
    run_extraction_batch(args.directory, args.currency, target_lang)

if __name__ == "__main__":
    main()