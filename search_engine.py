"""
Core search module for Nepali Hybrid Search (BM25 + Dense Vector) on Elasticsearch.
"""

import os
import builtins

# Global UTF-8 encoding patch for Windows environment compatibility
_orig_open = builtins.open

def _utf8_open(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
    if isinstance(mode, str) and "b" not in mode and encoding is None:
        encoding = "utf-8"
    return _orig_open(file, mode=mode, buffering=buffering, encoding=encoding, errors=errors, newline=newline, closefd=closefd, opener=opener)

builtins.open = _utf8_open

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["PYTHONUTF8"] = "1"

import re
import unicodedata
from typing import Dict, List, Any, Optional, Tuple

from elasticsearch import Elasticsearch, helpers
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from nepali_stemmer.stemmer import NepStemmer
from nepalikit.manage_stopwords import get_stopwords

STOPWORDS = set(get_stopwords())
STEMMER = NepStemmer()


def preprocess_nepali_text(text: str, return_steps: bool = False) -> Any:
    """
    Preprocess Nepali Devanagari text:
    1. Unicode normalization (NFKC) & zero-width joiner removal
    2. Strip non-Devanagari noise & HTML tags
    3. Filter stop words
    4. Suffix stripping via NepStemmer
    """
    if not text:
        if return_steps:
            return {
                "raw": "",
                "nfkc": "",
                "symbols_stripped": "",
                "stopwords_removed": [],
                "removed_stopwords": [],
                "stemmed_tokens": [],
                "final": "",
            }
        return ""

    # 1. Unicode normalization (NFKC) & zero-width joiner removal
    nfkc_text = unicodedata.normalize("NFKC", text)
    nfkc_text = nfkc_text.replace("\u200d", "").replace("\u200c", "")

    # 2. Noise & symbol stripping (keep Devanagari range U+0900-U+097F and whitespace)
    symbols_stripped = re.sub(r"<[^>]+>", "", nfkc_text)
    symbols_stripped = re.sub(r"[^\u0900-\u097F\s]", "", symbols_stripped)

    # 3. Stopword filtering
    all_tokens = symbols_stripped.split()
    tokens = [word for word in all_tokens if word not in STOPWORDS]
    removed_stopwords = [word for word in all_tokens if word in STOPWORDS]

    # 4. Stemming
    if not tokens:
        final_str = ""
        stemmed_tokens = []
    else:
        stemmed_str = STEMMER.stem(" ".join(tokens))
        stemmed_tokens = stemmed_str.split()
        final_str = stemmed_str

    if return_steps:
        return {
            "raw": text,
            "nfkc": nfkc_text,
            "symbols_stripped": symbols_stripped,
            "stopwords_removed": tokens,
            "removed_stopwords": removed_stopwords,
            "stemmed_tokens": stemmed_tokens,
            "final": final_str,
        }

    return final_str


def get_es_client(es_url: str = "http://localhost:9200") -> Elasticsearch:
    """Instantiate an Elasticsearch client."""
    return Elasticsearch(
        es_url,
        request_timeout=60,
        max_retries=3,
        retry_on_timeout=True,
    )


def check_es_connection(es: Elasticsearch) -> bool:
    """Return True if Elasticsearch server is reachable."""
    try:
        return es.ping()
    except Exception:
        return False


def load_encoder(model_name: str = "intfloat/multilingual-e5-large") -> SentenceTransformer:
    """Load the sentence transformer model for dense vector embeddings."""
    return SentenceTransformer(model_name)


def build_index(es: Elasticsearch, index_name: str, dims: int = 1024) -> None:
    """Create or overwrite an Elasticsearch index with BM25 + dense_vector mapping."""
    index_mapping = {
        "mappings": {
            "properties": {
                "title": {
                    "type": "text",
                    "analyzer": "whitespace",
                },
                "content": {
                    "type": "text",
                    "analyzer": "whitespace",
                },
                "raw_text": {
                    "type": "text",
                    "index": False,
                },
                "embedding": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        }
    }

    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    es.indices.create(index=index_name, body=index_mapping)


def generate_actions(index_name: str, docs: List[Dict[str, Any]], embeddings: List[List[float]]):
    for row, embedding in zip(docs, embeddings):
        yield {
            "_index": index_name,
            "_id": str(row.get("doc_id", row.get("id"))),
            "_source": {
                "title": str(row.get("title", f"Document #{row.get('doc_id', row.get('id'))}")),
                "content": row["_cleaned_text"],
                "raw_text": str(row.get("articlebody", row.get("text", "")))[:1000],
                "embedding": embedding,
            },
        }


def index_dataset(
    es: Elasticsearch,
    encoder: SentenceTransformer,
    index_name: str = "nepai_ir_corpus",
    sample_size: int = 1000,
    batch_size: int = 32,
    progress_callback=None,
) -> Tuple[int, str]:
    """
    Stream documents from dataset.csv, preprocess, embed, and index into ES.
    """
    build_index(es, index_name)

    if progress_callback:
        progress_callback(0.1, "Loading dataset from dataset.csv...")

    import pandas as pd
    try:
        if sample_size and sample_size > 0:
            df = pd.read_csv("dataset.csv", nrows=sample_size)
        else:
            df = pd.read_csv("dataset.csv")
        sample_docs = df.to_dict('records')
    except Exception as e:
        if progress_callback:
            progress_callback(1.0, f"Error reading dataset.csv: {e}")
        return 0, index_name

    if progress_callback:
        progress_callback(0.3, f"Preprocessing {len(sample_docs)} documents...")

    for row in sample_docs:
        text = str(row.get("articlebody", ""))
        if not text or text.lower() == 'nan':
            text = ""
        row["_cleaned_text"] = preprocess_nepali_text(text)

    if progress_callback:
        progress_callback(0.5, "Generating vector embeddings...")

    passages = [f"passage: {row['_cleaned_text']}" for row in sample_docs]
    embeddings = encoder.encode(passages, batch_size=batch_size, show_progress_bar=False).tolist()

    if progress_callback:
        progress_callback(0.8, "Bulk indexing into Elasticsearch...")

    success_count, _ = helpers.bulk(es, generate_actions(index_name, sample_docs, embeddings))

    if progress_callback:
        progress_callback(1.0, f"Indexing complete. Indexed {success_count} documents.")

    return success_count, index_name


def get_index_stats(es: Elasticsearch, index_name: str) -> Dict[str, Any]:
    """Retrieve total document count and health stats for the given ES index."""
    if not check_es_connection(es):
        return {"exists": False, "count": 0, "status": "ES Disconnected"}

    try:
        if not es.indices.exists(index=index_name):
            return {"exists": False, "count": 0, "status": "Index Not Found"}

        count_res = es.count(index=index_name)
        return {
            "exists": True,
            "count": count_res.get("count", 0),
            "status": "Ready",
        }
    except Exception as e:
        return {"exists": False, "count": 0, "status": f"Error: {str(e)}"}


def perform_search(
    es: Elasticsearch,
    encoder: SentenceTransformer,
    index_name: str,
    query_text: str,
    top_k: int = 5,
    mode: str = "hybrid",
    bm25_boost: float = 0.5,
    vector_boost: float = 0.5,
) -> Tuple[List[Dict[str, Any]], str, Dict[str, float]]:
    """
    Perform search on Elasticsearch index.
    Supported modes:
        - "hybrid": BM25 match + kNN dense_vector search combined
        - "lexical": BM25 match on text content only
        - "vector": kNN vector search only
    Returns:
        (results_list, cleaned_query_string, latency_dict)
    """
    import time

    t_start = time.time()
    cleaned_query = preprocess_nepali_text(query_text)
    effective_query = cleaned_query if isinstance(cleaned_query, str) and cleaned_query.strip() else query_text
    t_prep = time.time()

    search_body: Dict[str, Any] = {"size": top_k, "_source": ["title", "content", "raw_text"]}

    encoding_time_ms = 0.0

    if mode in ("hybrid", "vector"):
        t_enc_start = time.time()
        query_embedding = encoder.encode(f"query: {effective_query}").tolist()
        encoding_time_ms = (time.time() - t_enc_start) * 1000

        knn_dict: Dict[str, Any] = {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": top_k,
            "num_candidates": max(50, top_k * 5),
        }
        if mode == "hybrid":
            knn_dict["boost"] = vector_boost
        search_body["knn"] = knn_dict

    if mode in ("hybrid", "lexical"):
        match_dict: Dict[str, Any] = {
            "query": effective_query,
        }
        if mode == "hybrid":
            match_dict["boost"] = bm25_boost
        search_body["query"] = {
            "match": {
                "content": match_dict
            }
        }

    t_es_start = time.time()
    response = es.search(index=index_name, **search_body)
    t_es_end = time.time()

    prep_time_ms = (t_prep - t_start) * 1000
    es_time_ms = (t_es_end - t_es_start) * 1000
    total_time_ms = (t_es_end - t_start) * 1000

    latency_dict = {
        "prep_ms": prep_time_ms,
        "encoding_ms": encoding_time_ms,
        "es_ms": es_time_ms,
        "total_ms": total_time_ms,
    }

    results = []
    for hit in response.get("hits", {}).get("hits", []):
        src = hit.get("_source", {})
        results.append(
            {
                "id": hit["_id"],
                "score": hit["_score"],
                "title": src.get("title", f"Document #{hit['_id']}"),
                "content": src.get("content", ""),
                "raw_text": src.get("raw_text", src.get("content", "")),
            }
        )

    return results, cleaned_query if isinstance(cleaned_query, str) else cleaned_query.get("final", ""), latency_dict

