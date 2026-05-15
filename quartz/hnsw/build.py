import argparse
import time
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from quartz.quartz_core import IndexReader
from quartz.hnsw.graph import HNSWGraph

def build_hnsw(index_dir: Path, hnsw_dir: Path):
    reader = IndexReader(str(index_dir))
    num_docs = reader.num_docs() if hasattr(reader, "num_docs") else 0
    if num_docs == 0:
        print("No documents found in index.")
        return

    print("Loading SentenceTransformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    dim = model.get_sentence_embedding_dimension()

    print(f"Building HNSW graph for {num_docs} documents (dim={dim})...")
    graph = HNSWGraph(dim=dim)

    batch_size = 64
    docs_to_process = []

    # Iterate over all document IDs in the index
    for doc_id in tqdm(range(num_docs), desc="Embedding"):
        meta = reader.get_doc_meta(doc_id)
        # Ideally, we should embed the document text. The IndexReader API
        # might not expose raw text directly, so we use URL/title or a placeholder
        # In a real app, you'd retrieve the text from a forward index or KV store.
        # For demonstration, we'll embed the URL if text is missing.
        text = meta.get("url", f"doc_{doc_id}") 
        docs_to_process.append((doc_id, text))

        if len(docs_to_process) >= batch_size:
            texts = [t for _, t in docs_to_process]
            embeddings = model.encode(texts, normalize_embeddings=True)
            for (did, _), vec in zip(docs_to_process, embeddings):
                graph.add(did, vec)
            docs_to_process = []

    if docs_to_process:
        texts = [t for _, t in docs_to_process]
        embeddings = model.encode(texts, normalize_embeddings=True)
        for (did, _), vec in zip(docs_to_process, embeddings):
            graph.add(did, vec)

    print(f"Saving graph to {hnsw_dir}...")
    graph.save(hnsw_dir)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default="data/index/")
    parser.add_argument("--hnsw-dir", default="data/hnsw/")
    args = parser.parse_args()
    build_hnsw(Path(args.index_dir), Path(args.hnsw_dir))
