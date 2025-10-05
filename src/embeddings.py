import os, json
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import faiss
from tqdm import tqdm

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_text_from_lines(markdown_text: str, start_line: int, end_line: int) -> str:
    lines = markdown_text.splitlines()
    return "\n".join(lines[start_line-1:end_line + 1])

def build_section_embeddings(jsonl_file: str, markdown_file: str, output_prefix: str = None):
    """
    Build embeddings for each section of a Markdown document based on JSONL metadata.
    Saves FAISS index and metadata for fast semantic retrieval.
    """
    # Create embeddings directory
    embeddings_dir = Path("data/embeddings")
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # Set default output prefix based on markdown file if not provided
    if output_prefix is None:
        doc_name = Path(markdown_file).stem
        output_prefix = embeddings_dir / doc_name
    else:
        output_prefix = embeddings_dir / Path(output_prefix).name

    # Check if embeddings already exist
    faiss_file = f"{output_prefix}.faiss"
    npz_file = f"{output_prefix}.npz"
    
    if Path(faiss_file).exists() and Path(npz_file).exists():
        print(f"✅ Embeddings already exist at {output_prefix}.faiss/.npz")
        return

    # Load markdown
    with open(markdown_file, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    # Load JSONL
    sections = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            sections.append(json.loads(line.strip()))

    texts = []
    metadata = []
    
    merge_next = 3
    
    i = 0
    while i < len(sections):
        sec = sections[i]
        start, end = sec["lines"]
        section_text = get_text_from_lines(markdown_text, start, end).strip()

        # Check if section is strictly one line
        if end - start == 0:
            merged_text = section_text
            merged_count = 0
            merged_start = start
            merged_end = end

            # Merge next few sections (up to merge_next)
            for j in range(1, merge_next + 1):
                if i + j < len(sections):
                    next_start, next_end = sections[i + j]["lines"]
                    next_text = get_text_from_lines(markdown_text, next_start, next_end)
                    merged_text += "\n\n" + next_text
                    merged_end = next_end  # update merged_end dynamically
                    merged_count += 1

            section_text = merged_text
            print(f"⚙️ Merged one-line section '{sec['title']}' with next {merged_count} sections "
                f"({merged_start}-{merged_end}).")

            # Skip over the merged sections
            i += merged_count + 1
        else:
            merged_start, merged_end = start, end
            i += 1

        texts.append(section_text)
        metadata.append({
            "section_id": sec["section_id"],
            "title": sec["title"],
            "section_num": sec["section_number"],
            "lines": [merged_start, merged_end],
            "lang": sec["lang"],
            "char_count": len(section_text)
        })

    print(f"Building embeddings for {len(texts)} sections...")

    embeddings = []
    for text in tqdm(texts):
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embeddings.append(response.data[0].embedding)

    def normalize(vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms

    embeddings = np.array(embeddings).astype("float32")
    embeddings = normalize(embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, f"{output_prefix}.faiss")
    np.savez(f"{output_prefix}.npz", metadata=metadata)
    print(f"✅ Saved FAISS index and metadata to {output_prefix}.faiss / .npz")
    
    
def search_sections(query: str, top_k: int = 5, md_file: str = None):
    """
    Search for the most semantically relevant sections to a query using FAISS.
    Returns top-k sections and their metadata.
    """
    
    # Load FAISS + metadata
    index = faiss.read_index(f"data/embeddings/{md_file}.faiss")
    meta = np.load(f"data/embeddings/{md_file}.npz", allow_pickle = True)["metadata"]

    # Embed the query
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    def normalize(vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms
    
    q_vec = normalize(np.array([query_embedding]).astype("float32"))
    distances, indices = index.search(q_vec, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        results.append({
            "rank": rank + 1,
            "title": meta[idx]["title"],
            "section_id": meta[idx]["section_id"],
            "lines": meta[idx]["lines"],
            "char_count": meta[idx]["char_count"],
            "distance": float(distances[0][rank])
        })
    return results
    
    
def append_next_sections(md_file: str, current_section_id: str, num_next: int = 5) -> str:
    
    meta = np.load(f"data/embeddings/{md_file}.npz", allow_pickle=True)["metadata"]
    with open(f"data/parsed/{md_file}.md", "r", encoding="utf-8") as f:
        markdown_text = f.read()

    # Find current section index in metadata
    current_idx = next(
        (i for i, m in enumerate(meta) if m["section_id"] == current_section_id),
        None
    )
    if current_idx is None:
        return ""

    # Start with current section’s text
    start_line, end_line = meta[current_idx]["lines"]
    combined_text = get_text_from_lines(markdown_text, start_line, end_line)

    for offset in range(1, num_next + 1):
        next_idx = current_idx + offset
        if next_idx < len(meta):
            next_start, next_end = meta[next_idx]["lines"]
            next_text = get_text_from_lines(markdown_text, next_start, next_end)
            combined_text += "\n\n" + next_text

    return combined_text




if __name__ == "__main__":
    # Build embeddings and save to data/embeddings/ folder
    # build_section_embeddings("data/sections_report/nvidia_form_10-k_2023_parsed.jsonl", "data/parsed/nvidia_form_10-k_2023_parsed.md")
    build_section_embeddings("data/sections_report/chemming_raw_parsed.jsonl", "data/parsed/chemming_raw_parsed.md")

    # Test search functionality
    print("\n" + "="*50)
    print("TESTING SEARCH FUNCTIONALITY")
    print("="*50)

    # meta = np.load("data/embeddings/nvidia_form_10-k_2023_parsed.npz", allow_pickle=True)
    meta = np.load("data/embeddings/chemming_raw_parsed.npz", allow_pickle=True)
    metadata = meta["metadata"]
    
    # Example searches
    
    test_queries = [
        "company headquarters location office"
    ]
    
    #with open("data/parsed/nvidia_form_10-k_2023_parsed.md", "r", encoding="utf-8") as f:
    #  markdown_text = f.read()
    with open("data/parsed/chemming_raw_parsed.md", "r", encoding="utf-8") as f:
      markdown_text = f.read()
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        results = search_sections(query, top_k=3, md_file="chemming_raw_parsed")
        for result in results:
            print(f"{result['rank']}. {result['title']} (distance: {result['distance']:.3f})")
            print(f"   Section ID: {result['section_id']}")
            print(f"   Lines: {result['lines'][0]}-{result['lines'][1]}")
            print(f"   Chars: {result['char_count']:,}")
            print(f"   Original text:\n {get_text_from_lines(markdown_text, result['lines'][0], result['lines'][1])}")
