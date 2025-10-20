#!/usr/bin/env python3
"""
Test script for the updated extraction.py with dual markdown file support
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_extraction_dual_files():
    """Test extraction with both 2024 and 2023 markdown files"""
    print("🧪 Testing Dual-File Extraction")
    print("=" * 40)
    
    # Check for sample markdown files
    data_dir = Path("data/parsed")
    if not data_dir.exists():
        print("❌ data/parsed directory not found")
        return False
    
    # Find sample markdown files
    md_files = list(data_dir.glob("*_raw_parsed.md"))
    if len(md_files) < 1:
        print("❌ No markdown files found in data/parsed")
        return False
    
    print(f"📁 Found {len(md_files)} markdown files:")
    for i, file in enumerate(md_files[:5]):  # Show first 5
        print(f"   {i+1}. {file.name}")
    
    # Use the first available file as both 2024 and 2023 for testing
    test_file_2024 = str(md_files[0])
    test_file_2023 = str(md_files[1]) if len(md_files) > 1 else None
    
    print(f"\n🔬 Testing extraction with:")
    print(f"   2024 file: {Path(test_file_2024).name}")
    if test_file_2023:
        print(f"   2023 file: {Path(test_file_2023).name}")
    else:
        print(f"   2023 file: None (will test single file mode)")
    
    try:
        from extraction import extract
        
        # Test the extraction
        report = extract(test_file_2024, test_file_2023)
        
        if report:
            print("\n✅ Extraction successful!")
            print(f"   Company Name: {report.basic_info.company_name}")
            print(f"   Headquarters: {report.basic_info.headquarters_location}")
            print(f"   Mission: {report.mission_vision.mission_statement[:100] if report.mission_vision.mission_statement != 'N/A' else 'N/A'}...")
            return True
        else:
            print("\n❌ Extraction returned None")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Extraction error: {e}")
        return False

def test_extraction_single_file():
    """Test extraction with only 2024 markdown file"""
    print("\n🧪 Testing Single-File Extraction")
    print("=" * 40)
    
    data_dir = Path("data/parsed")
    md_files = list(data_dir.glob("*_raw_parsed.md"))
    
    if len(md_files) < 1:
        print("❌ No markdown files found")
        return False
    
    test_file = str(md_files[0])
    print(f"📁 Testing with: {Path(test_file).name}")
    
    try:
        from extraction import extract
        
        # Test single file extraction
        report = extract(test_file, None)
        
        if report:
            print("✅ Single-file extraction successful!")
            return True
        else:
            print("❌ Single-file extraction returned None")
            return False
            
    except Exception as e:
        print(f"❌ Single-file extraction error: {e}")
        return False

def main():
    """Run extraction tests"""
    print("🔬 FinDDR Extraction Testing Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 2
    
    # Test 1: Dual file extraction
    if test_extraction_dual_files():
        tests_passed += 1
    
    # Test 2: Single file extraction
    if test_extraction_single_file():
        tests_passed += 1
    
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Extraction system is working correctly.")
        print("\n📋 You can now use:")
        print("   python src/main.py file1.pdf file2.pdf")
        print("   python src/extraction.py file_2024.md file_2023.md")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()