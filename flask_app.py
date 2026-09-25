import os
import logging
import tempfile
import pandas as pd
from flask import Flask, render_template, request, jsonify
from search_engine import (
    get_es_client,
    check_es_connection,
    load_encoder,
    perform_search,
    get_index_stats,
    index_dataset,
    find_duplicate_rows,
    preprocess_nepali_text,
)

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
INDEX_NAME = os.environ.get("INDEX_NAME", "nepai_ir_corpus")

# Application level globals (lazy loaded)
es_client = None
encoder = None

def get_es():
    global es_client
    if es_client is None:
        es_client = get_es_client(ES_URL)
    return es_client

def get_enc():
    global encoder
    if encoder is None:
        logger.info("Loading embedding model...")
        encoder = load_encoder("intfloat/multilingual-e5-large")
        logger.info("Model loaded.")
    return encoder

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/search", methods=["POST"])
def search_api():
    data = request.json
    query = data.get("query", "")
    mode = data.get("mode", "hybrid")
    top_k = int(data.get("top_k", 5))
    bm25_boost = float(data.get("bm25_boost", 0.5))
    vector_boost = float(data.get("vector_boost", 0.5))

    if not query.strip():
        return jsonify({"error": "Empty query"}), 400

    es = get_es()
    if not check_es_connection(es):
        return jsonify({"error": "Elasticsearch is offline"}), 503

    enc = get_enc()
    
    try:
        results, cleaned_query, latency = perform_search(
            es,
            enc,
            index_name=INDEX_NAME,
            query_text=query,
            top_k=top_k,
            mode=mode,
            bm25_boost=bm25_boost,
            vector_boost=vector_boost,
        )
        return jsonify({
            "results": results,
            "cleaned_query": cleaned_query,
            "latency": latency
        })
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/compare", methods=["POST"])
def compare_api():
    data = request.json
    query = data.get("query", "")
    top_k = int(data.get("top_k", 5))
    bm25_boost = float(data.get("bm25_boost", 0.5))
    vector_boost = float(data.get("vector_boost", 0.5))

    if not query.strip():
        return jsonify({"error": "Empty query"}), 400

    es = get_es()
    if not check_es_connection(es):
        return jsonify({"error": "Elasticsearch is offline"}), 503

    enc = get_enc()
    try:
        res_hybrid, _, lat_hybrid = perform_search(es, enc, INDEX_NAME, query, top_k, "hybrid", bm25_boost, vector_boost)
        res_lexical, _, lat_lexical = perform_search(es, enc, INDEX_NAME, query, top_k, "lexical")
        res_vector, _, lat_vector = perform_search(es, enc, INDEX_NAME, query, top_k, "vector")

        return jsonify({
            "hybrid": {"results": res_hybrid, "latency": lat_hybrid},
            "lexical": {"results": res_lexical, "latency": lat_lexical},
            "vector": {"results": res_vector, "latency": lat_vector}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/diagnostics", methods=["GET"])
def diagnostics_api():
    es = get_es()
    es_connected = check_es_connection(es)
    
    diagnostics = {
        "es_connected": es_connected,
        "index_name": INDEX_NAME,
        "cluster_info": None,
        "index_stats": None
    }
    
    if es_connected:
        try:
            diagnostics["cluster_info"] = dict(es.info())
            diagnostics["index_stats"] = get_index_stats(es, INDEX_NAME)
        except Exception as e:
            logger.error(f"Failed to get diagnostic info: {e}")
            
    return jsonify(diagnostics)

@app.route("/api/reindex", methods=["POST"])
def reindex_api():
    data = request.json
    sample_size = int(data.get("sample_size", 500))

    try:
        es = get_es()
        if not check_es_connection(es):
            return jsonify({"error": "Elasticsearch is offline"}), 503

        enc = get_enc()
        count, _ = index_dataset(
            es, 
            enc, 
            index_name=INDEX_NAME, 
            sample_size=sample_size, 
            progress_callback=None
        )
        return jsonify({"success": True, "indexed_count": count})
    except Exception as e:
        logger.error(f"Re-indexing failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/upload_index", methods=["POST"])
def upload_index_api():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file and file.filename.endswith('.csv'):
        temp_path = None
        filtered_path = None
        try:
            fd, temp_path = tempfile.mkstemp(suffix=".csv")
            os.close(fd)
            file.save(temp_path)
            
            # Check schema inconsistency early on
            df = pd.read_csv(temp_path, nrows=1)
            columns = df.columns.tolist()
            
            has_id = 'id' in columns or 'doc_id' in columns
            has_text = 'text' in columns or 'articlebody' in columns
            
            if not has_id or not has_text:
                os.remove(temp_path)
                return jsonify({"error": "Schema inconsistency: CSV must contain 'id'/'doc_id' and 'text'/'articlebody' columns."}), 400
            
            es = get_es()
            if not check_es_connection(es):
                os.remove(temp_path)
                return jsonify({"error": "Elasticsearch is offline"}), 503

            docs, duplicate_count = find_duplicate_rows(
                es, INDEX_NAME, df.to_dict("records")
            )
            if not docs:
                os.remove(temp_path)
                return jsonify({
                    "success": True,
                    "indexed_count": 0,
                    "duplicate_count": duplicate_count,
                })

            filtered_fd, filtered_path = tempfile.mkstemp(suffix=".csv")
            os.close(filtered_fd)
            pd.DataFrame(docs).to_csv(filtered_path, index=False)

            enc = get_enc()

            count, _ = index_dataset(
                es, 
                enc, 
                index_name=INDEX_NAME, 
                sample_size=0, 
                progress_callback=None,
                overwrite=False,
                file_path=filtered_path,
                op_type="create",
            )

            os.remove(temp_path)
            os.remove(filtered_path)
            return jsonify({
                "success": True,
                "indexed_count": count,
                "duplicate_count": duplicate_count,
            })
            
        except Exception as e:
            logger.error(f"Incremental indexing failed: {e}")
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            if filtered_path and os.path.exists(filtered_path):
                os.remove(filtered_path)
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Only CSV files are allowed"}), 400

@app.route("/api/nlp_sandbox", methods=["POST"])
def nlp_sandbox_api():
    data = request.json
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"steps": {}})
        
    steps = preprocess_nepali_text(text, return_steps=True)
    return jsonify({"steps": steps})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
