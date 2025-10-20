import numpy as np

meta = np.load("data/embeddings/nvidia_2023_raw_parsed.npz", allow_pickle=True)
metadata = meta["metadata"]

print(f"Loaded {len(metadata)} sections\n")

for i, section in enumerate(metadata, start=1):
    print(f"--- Section {i} ---")
    for key, value in section.items():
        print(f"{key}: {value}")
    print()  # blank line between sections
