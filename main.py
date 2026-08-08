"""
CLI runner for Nepali hybrid search prototype (BM25 + dense vector) on Elasticsearch.
"""

import sys
from search_engine import (
    get_es_client,
    check_es_connection,
    load_encoder,
    index_dataset,
    perform_search,
    get_index_stats,
)


def main():
    es_url = "http://localhost:9200"
    index_name = "nepali_wikipedia_prototype"

    print("Connecting to Elasticsearch...")
    es = get_es_client(es_url)
    if not check_es_connection(es):
        sys.exit(
            f"Could not reach Elasticsearch at {es_url} — "
            "make sure it's running before executing this script."
        )

    print("Loading embedding model...")
    encoder = load_encoder("intfloat/multilingual-e5-large")

    print("Checking dataset indexing status...")
    stats = get_index_stats(es, index_name)
    if not stats.get("exists") or stats.get("count", 0) == 0:
        print("Index empty or missing. Indexing sample dataset...")
        index_dataset(es, encoder, index_name=index_name, sample_size=1000)
    else:
        print(f"Index '{index_name}' ready with {stats['count']} documents.")

    sample_query = "नेपालको इतिहास र संस्कृति"
    print(f"\nPerforming Hybrid Search for: '{sample_query}'...")
    results, cleaned, latency = perform_search(
        es, encoder, index_name, sample_query, top_k=5, mode="hybrid"
    )

    print(f"\n--- Top {len(results)} Results for: '{sample_query}' ---")
    print(f"Cleaned/Stemmed Query: '{cleaned}'")
    print(f"Latency: {latency['total_ms']:.2f} ms (Prep: {latency['prep_ms']:.2f}ms, Enc: {latency['encoding_ms']:.2f}ms, ES: {latency['es_ms']:.2f}ms)\n")
    for r in results:
        print(f"ID: {r['id']} | Title: {r.get('title', '')} | Score: {r['score']:.4f}")
        print(f"Text: {r['content'][:150]}...\n")


if __name__ == "__main__":
    main()