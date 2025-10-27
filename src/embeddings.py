import os, json
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError
import numpy as np
import faiss
from tqdm import tqdm
import time
import random
import tiktoken
from typing import List


load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def retrieve_relevant_text(search_queries: List[str], top_k: int, md_file: str) -> str:
    """
    Search for sections using multiple queries and return the complete combined text.
    Handles duplicate section names by using composite keys (section_id + line_range).
    
    Args:
        search_queries: List of search query strings
        top_k: Number of top results to return per query
        md_file: Name of the markdown file (without .md extension)
    
    Returns:
        Combined text from all relevant sections
    """
    all_results = []
    for query in search_queries:
        # returns list of dict, where each item is section identifier  
        results = search_sections(query, top_k=top_k, md_file=md_file)
        all_results.extend(results)
    
    # Sort by relevance (distance score) first
    all_results.sort(key=lambda x: x.get('distance', float('inf')))
    
    # De-duplicate using composite key: section_id + line_range + section_number
    seen_sections = set()
    unique_results = []
    
    for result in all_results:
        # Create composite key to handle duplicate section names
        start_line, end_line = result["lines"]
        composite_key = f"{result['section_id']}_{start_line}_{end_line}_{result.get('section_number', 0)}"
        
        if composite_key not in seen_sections:
            seen_sections.add(composite_key)
            unique_results.append(result)
  
    print(f"Final unique results: {len(unique_results)} sections")
    
    # Get the actual text for selected sections
    with open(f"data/parsed_md_val_mistral/{md_file}.md", "r", encoding="utf-8") as f:
        markdown_text = f.read()
    
    lines = markdown_text.split('\n')
    context = ""

    for h in unique_results:
        s, e = h["lines"]
        section_text = '\n'.join(lines[s - 1:e + 1])
        # Include section number for clarity when there are duplicate titles
        section_identifier = f"{h.get('title')}"
        context += f"\n--- {section_identifier} ---\n{section_text}\n\n"
    
    return context.strip()

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
        print(f"      ✅ Embeddings already exist at {output_prefix}.faiss/.npz")
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
            print(f" Merged one-line section '{sec['title']}' with next {merged_count} sections "
                f"({merged_start}-{merged_end}).")

            # Skip over the merged sections
            i += merged_count + 1
        else:
            merged_start, merged_end = start, end
            i += 1

        section_title = sec.get("title", "")
        if section_title:
            section_text = f"{section_title}\n\n{section_text}"
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

    enc = tiktoken.encoding_for_model("text-embedding-3-large")
    MAX_TOKENS = 8000       
    CHUNK_OVERLAP = 128     

    def _embed_chunk(text: str) -> np.ndarray:
        for attempt in range(5):
            try:
                resp = client.embeddings.create(model="text-embedding-3-large", input=text)
                return np.array(resp.data[0].embedding, dtype=np.float32)
            except (RateLimitError, APIError) as e:
                wait = 2 ** attempt + random.random()
                print(f"[warn] Retry {attempt+1}: waiting {wait:.1f}s ({e})")
                time.sleep(wait)
        raise RuntimeError("Failed to embed after retries.")

    def _chunk_by_tokens(text: str, max_tokens: int = MAX_TOKENS, overlap: int = CHUNK_OVERLAP) -> list[str]:
        toks = enc.encode(text)
        if len(toks) <= max_tokens:
            return [text]
        chunks = []
        start = 0
        while start < len(toks):
            end = min(start + max_tokens, len(toks))
            chunk = enc.decode(toks[start:end])
            chunks.append(chunk)
            if end == len(toks):
                break
            # move with overlap
            start = end - overlap if end - overlap > start else end
        return chunks

    oversized = sum(1 for m in metadata if m["char_count"] > 20000)
    if oversized:
        print(f"[info] Detected {oversized} very large sections (>20k chars). Chunking will apply.")

    embeddings = []
    for text in tqdm(texts):
        chunks = _chunk_by_tokens(text)
        if len(chunks) == 1:
            emb = _embed_chunk(chunks[0])
        else:
            sub = [_embed_chunk(c) for c in chunks]
            emb = np.mean(np.vstack(sub), axis=0).astype(np.float32)
            print(f"[warn] Chunked long section into {len(chunks)} parts (avg pooled).")
        embeddings.append(emb)


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
        model="text-embedding-3-large",
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
            "section_number": meta[idx]["section_num"],
            "section_id": meta[idx]["section_id"],
            "lines": meta[idx]["lines"],
            "char_count": meta[idx]["char_count"],
            "distance": float(distances[0][rank])
        })
    return results
    
    
def append_next_sections(md_file: str, current_section_id: str, num_next: int = 5) -> str:
    
    meta = np.load(f"data/embeddings/{md_file}.npz", allow_pickle=True)["metadata"]
    with open(f"data/parsed_md_val_mistral/{md_file}.md", "r", encoding="utf-8") as f:
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
    import sys
    from pathlib import Path

    parsed_dir = Path("data/parsed_md_val_mistral")
    sections_dir = Path("data/sections_report")
    embeddings_dir = Path("data/embeddings")
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted(parsed_dir.glob("*_mistral.md"))
    if not md_files:
        print(f"❌ No markdown files found in {parsed_dir}")
        sys.exit(1)

    print(f"📄 Found {len(md_files)} markdown files for embedding.\n")

    for md_path in md_files:
        base_name = md_path.stem
        jsonl_path = sections_dir / f"{base_name}.jsonl"
        faiss_path = embeddings_dir / f"{base_name}.faiss"
        npz_path = embeddings_dir / f"{base_name}.npz"

        if not jsonl_path.exists():
            print(f"⚠️  Skipping {md_path.name} — JSONL missing: {jsonl_path}")
            continue

        if faiss_path.exists() and npz_path.exists():
            print(f"✅ Skipping {base_name} (embeddings already exist)\n")
            continue

        # Print both absolute paths clearly
        print(f"\n🚀 Building embeddings for {base_name} ...")
        print(f"   📘 Markdown: {md_path.resolve()}")
        print(f"   📄 JSONL:    {jsonl_path.resolve()}")

        try:
            build_section_embeddings(str(jsonl_path), str(md_path))
            print(f"✅ Completed {base_name}\n")
        except Exception as e:
            print(f"❌ Error processing {base_name}: {e}\n")

    print("🎉 All embeddings finished successfully!")

    
    
# if __name__ == "__main__":
#     # Build embeddings and save to data/embeddings/ folder
#     build_section_embeddings("data/sections_report/nvidia_2023_raw_parsed.jsonl", "data/parsed/nvidia_2023_raw_parsed.md")

#     # Test search functionality
#     print("\n" + "="*50)
#     print("TESTING SEARCH FUNCTIONALITY")
#     print("="*50)

#     meta = np.load("data/embeddings/nvidia_2023_raw_parsed.npz", allow_pickle=True)
#     # meta = np.load("data/embeddings/catl_2024_raw_parsed.npz", allow_pickle=True)
#     metadata = meta["metadata"]
    
#     search_queries = [
#             "core values",
#             "mission statement", "company mission statement", "our mission", "mission",
#             "vision statement", "company vision statement", "our vision", "vision future",
#             "innovation mission vision values", "integrity", "collaboration", "diversity inclusion",
#             "core values principles", "corporate values", "company values beliefs"
#     ]

#     with open("data/parsed/nvidia_2023_raw_parsed.md", "r", encoding="utf-8") as f:
#         markdown_text = f.read()
    
#     combined_text = retrieve_relevant_text(search_queries, top_k=20, md_file="nvidia_2023_raw_parsed")
#     print(combined_text)

#     for query in search_queries:
#         print(f"\nQuery: '{query}'")
#         print("-" * 40)
#         results = search_sections(query, top_k=5, md_file="nvidia_2023_raw_parsed")
#         for result in results:
#             print(f"{result['rank']}. {result['title']} (distance: {result['distance']:.3f})")
#             print(f"   Section ID: {result['section_id']}")
#             print(f"   Lines: {result['lines'][0]}-{result['lines'][1]}")
#             print(f"   Chars: {result['char_count']:,}")
#             # print(f"   Original text:\n {get_text_from_lines(markdown_text, result['lines'][0], result['lines'][1])}")
