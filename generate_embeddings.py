#!/usr/bin/env python3
"""
Pre-compute embeddings for pack.json chunks.

Usage:
    python generate_embeddings.py --input pack.json --output pack_with_embeddings.json

Requires:
    - AR_URL environment variable (e.g., https://maxdemo.staging.answerrocket.com)
    - AR_TOKEN environment variable (API token)
"""

import json
import argparse
import os
import sys
from answer_rocket import AnswerRocketClient

BATCH_SIZE = 50


def load_pack(input_file):
    """Load pack.json file"""
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pack(data, output_file):
    """Save pack.json file with embeddings"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def generate_embeddings(pack_data):
    """Generate embeddings for all chunks in pack.json"""

    # Initialize client
    ar_client = AnswerRocketClient()

    # Handle different pack formats
    if isinstance(pack_data, list):
        documents = pack_data
    elif isinstance(pack_data, dict) and "Documents" in pack_data:
        documents = pack_data["Documents"]
    else:
        raise ValueError(f"Unexpected pack format: {type(pack_data)}")

    # Collect all chunks with their locations
    all_chunks = []
    for doc_idx, doc in enumerate(documents):
        file_name = doc.get("File", "unknown")
        chunks = doc.get("Chunks", [])
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append({
                "doc_idx": doc_idx,
                "chunk_idx": chunk_idx,
                "text": chunk.get("Text", ""),
                "file_name": file_name
            })

    print(f"Found {len(all_chunks)} chunks across {len(documents)} documents")

    # Generate embeddings in batches
    total_batches = (len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num, i in enumerate(range(0, len(all_chunks), BATCH_SIZE)):
        batch = all_chunks[i:i + BATCH_SIZE]
        batch_texts = [c["text"] for c in batch]

        print(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch)} chunks)...")

        response = ar_client.llm.generate_embeddings(batch_texts)

        # Check for errors
        if hasattr(response, 'success') and not response.success:
            error_msg = getattr(response, 'error', 'Unknown error')
            raise Exception(f"Embedding API failed: {error_msg}")

        # Extract embeddings
        embeddings = []
        if hasattr(response, 'embeddings') and response.embeddings:
            for item in response.embeddings:
                if hasattr(item, 'vector'):
                    embeddings.append(item.vector)
                elif hasattr(item, 'embedding'):
                    embeddings.append(item.embedding)
                elif isinstance(item, list):
                    embeddings.append(item)
        elif isinstance(response, list):
            embeddings = response

        if len(embeddings) != len(batch):
            raise Exception(f"Expected {len(batch)} embeddings, got {len(embeddings)}")

        # Store embeddings back in pack data
        for j, chunk_info in enumerate(batch):
            doc_idx = chunk_info["doc_idx"]
            chunk_idx = chunk_info["chunk_idx"]
            documents[doc_idx]["Chunks"][chunk_idx]["Embedding"] = embeddings[j]

    print(f"Successfully generated {len(all_chunks)} embeddings")

    # Return updated pack data
    if isinstance(pack_data, dict):
        pack_data["Documents"] = documents
        return pack_data
    else:
        return documents


def main():
    parser = argparse.ArgumentParser(description="Pre-compute embeddings for pack.json")
    parser.add_argument("--input", "-i", required=True, help="Input pack.json file")
    parser.add_argument("--output", "-o", required=True, help="Output file with embeddings")
    args = parser.parse_args()

    # Check environment
    if not os.environ.get("AR_URL"):
        print("Warning: AR_URL not set. Using default.")

    # Load, process, save
    print(f"Loading {args.input}...")
    pack_data = load_pack(args.input)

    print("Generating embeddings...")
    pack_with_embeddings = generate_embeddings(pack_data)

    print(f"Saving to {args.output}...")
    save_pack(pack_with_embeddings, args.output)

    print("Done!")


if __name__ == "__main__":
    main()
