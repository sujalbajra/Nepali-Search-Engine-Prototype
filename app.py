import os
import builtins

# Force UTF-8 text file opening across Windows environment
_orig_open = builtins.open

def _utf8_open(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
    if isinstance(mode, str) and "b" not in mode and encoding is None:
        encoding = "utf-8"
    return _orig_open(file, mode=mode, buffering=buffering, encoding=encoding, errors=errors, newline=newline, closefd=closefd, opener=opener)

builtins.open = _utf8_open
os.environ["PYTHONUTF8"] = "1"

import time
import streamlit as st
from search_engine import (
    get_es_client,
    check_es_connection,
    load_encoder,
    perform_search,
    get_index_stats,
    index_dataset,
    preprocess_nepali_text,
)

# ------------------------------------------------------------------------------
# 1. Page Configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Nepali Hybrid Search Engine",
    page_icon="🇳🇵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# 2. Custom CSS & Design System
# ------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', 'Noto Sans Devanagari', sans-serif;
    }
    
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 40%, #e52d27 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    
    .metric-card .val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e293b;
    }
    
    .metric-card .lbl {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .result-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
        transition: all 0.2s ease;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
        border-color: #cbd5e1;
    }
    
    .doc-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.4rem;
    }
    
    .score-badge {
        display: inline-block;
        background: #eff6ff;
        color: #1d4ed8;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        border: 1px solid #bfdbfe;
    }

    .rank-badge {
        display: inline-block;
        background: #3b82f6;
        color: #ffffff;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        margin-right: 0.5rem;
    }
    
    .prep-box {
        background-color: #0f172a;
        color: #38bdf8;
        border-left: 4px solid #3b82f6;
        padding: 0.9rem 1.1rem;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }

    .step-box {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 0.9rem;
        margin-bottom: 0.8rem;
    }
    .step-title {
        font-weight: 700;
        color: #1e293b;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# 3. Model & ES Caching
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Embedding Model (multilingual-e5-large)...")
def get_cached_encoder():
    return load_encoder("intfloat/multilingual-e5-large")


@st.cache_resource(show_spinner="Connecting to Elasticsearch...")
def get_cached_es_client(url: str):
    return get_es_client(url)


# ------------------------------------------------------------------------------
# 4. Header & Sidebar Setup
# ------------------------------------------------------------------------------
st.markdown('<div class="main-title">🇳🇵 Nepali Hybrid Search Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Elasticsearch 8+ • BM25 Lexical • Dense Vector Embeddings (multilingual-e5-large) • Devanagari NLP Pipeline</div>',
    unsafe_allow_html=True,
)

st.sidebar.header("⚙️ Search Settings")

es_url = st.sidebar.text_input("Elasticsearch URL", value="http://localhost:9200")
index_name = "nepali_wikipedia_prototype"

# Connect to ES
es = get_cached_es_client(es_url)
es_connected = check_es_connection(es)

if es_connected:
    st.sidebar.success("🟢 Elasticsearch Connected")
else:
    st.sidebar.error("🔴 Elasticsearch Disconnected")
    st.error("Cannot connect to Elasticsearch at `http://localhost:9200`. Please start your Elasticsearch instance.")

# Retrieve index statistics
stats = get_index_stats(es, index_name)
st.sidebar.metric("Indexed Documents", value=stats.get("count", 0))

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Hybrid Search Tuning")

top_k = st.sidebar.slider("Top Results (k)", min_value=1, max_value=20, value=5)

bm25_boost = st.sidebar.slider("BM25 Lexical Boost", 0.0, 2.0, 0.5, step=0.1)
vector_boost = st.sidebar.slider("Dense Vector Boost", 0.0, 2.0, 0.5, step=0.1)

# Index Re-building Controls in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📦 Dataset Indexing")
sample_size = st.sidebar.number_input("Sample Dataset Size", min_value=100, max_value=5000, value=1000, step=100)

if st.sidebar.button("🔄 Re-index Wikipedia Dataset"):
    if not es_connected:
        st.sidebar.error("ES not connected!")
    else:
        progress_bar = st.sidebar.progress(0.0)
        status_text = st.sidebar.empty()

        def update_progress(pct, msg):
            progress_bar.progress(pct)
            status_text.text(msg)

        try:
            encoder = get_cached_encoder()
            count, _ = index_dataset(
                es, encoder, index_name=index_name, sample_size=sample_size, progress_callback=update_progress
            )
            st.sidebar.success(f"Indexed {count} docs!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Indexing failed: {e}")


# ------------------------------------------------------------------------------
# 5. Main Application Tabs
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔎 Single Query Search",
        "⚔️ Side-by-Side Mode Comparison",
        "📊 Index Health & Diagnostics",
        "🧪 Devanagari NLP Sandbox",
    ]
)

# ------------------------------------------------------------------------------
# TAB 1: Single Query Search
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("🔎 Nepali Devanagari Search")

    sample_queries = [
        "नेपालको इतिहास र संस्कृति",
        "गण्डकी प्रदेश",
        "संवैधानिक गणतन्त्र",
        "भारतीय वाङ्गमय र साहित्य",
    ]

    if "t1_user_query" not in st.session_state:
        st.session_state["t1_user_query"] = "नेपालको इतिहास र संस्कृति"

    st.markdown("**Quick Preset Queries:**")
    cols = st.columns(len(sample_queries))

    for idx, q in enumerate(sample_queries):
        if cols[idx].button(q, key=f"ex_t1_{idx}"):
            st.session_state["t1_user_query"] = q
            st.rerun()

    c1, c2 = st.columns([3, 1])
    with c1:
        user_query = st.text_input(
            "Enter search query in Nepali Devanagari:",
            key="t1_user_query",
            placeholder="उदा: नेपालको इतिहास र संस्कृति...",
        )
    with c2:
        search_mode_label = st.selectbox(
            "Search Engine Mode",
            options=[
                "⚡ Hybrid Search (BM25 + Dense Vector)",
                "🔤 Lexical Search Only (BM25)",
                "🧠 Dense Vector Search Only (Semantic)",
            ],
            index=0,
        )

    mode_map = {
        "⚡ Hybrid Search (BM25 + Dense Vector)": "hybrid",
        "🔤 Lexical Search Only (BM25)": "lexical",
        "🧠 Dense Vector Search Only (Semantic)": "vector",
    }
    selected_mode = mode_map[search_mode_label]

    if user_query.strip():
        if not es_connected:
            st.warning("Elasticsearch is offline. Please start your ES instance to execute search.")
        else:
            encoder = get_cached_encoder()

            results, cleaned_query, latency = perform_search(
                es,
                encoder,
                index_name=index_name,
                query_text=user_query,
                top_k=top_k,
                mode=selected_mode,
                bm25_boost=bm25_boost,
                vector_boost=vector_boost,
            )

            # Preprocessing Inspector Expander
            with st.expander("🛠️ Devanagari NLP Preprocessing Inspector", expanded=False):
                st.markdown(f"**Raw User Input:** `{user_query}`")
                st.markdown(f"**Cleaned, Filtered & NepStemmer Stemmed Query:**")
                st.markdown(
                    f'<div class="prep-box">{cleaned_query if cleaned_query else "(All terms filtered out or empty)"}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Pipeline: NFKC Normalization ➡️ Devanagari Symbol Filtering ➡️ Stopword Stripping ➡️ NepStemmer Suffix Stripping"
                )

            # Latency Telemetry Bar
            st.markdown("##### ⚡ Performance & Telemetry Breakdown")
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(
                f'<div class="metric-card"><div class="val">{latency["total_ms"]:.2f} ms</div><div class="lbl">Total Latency</div></div>',
                unsafe_allow_html=True,
            )
            m2.markdown(
                f'<div class="metric-card"><div class="val">{latency["prep_ms"]:.2f} ms</div><div class="lbl">NLP Preprocessing</div></div>',
                unsafe_allow_html=True,
            )
            m3.markdown(
                f'<div class="metric-card"><div class="val">{latency["encoding_ms"]:.2f} ms</div><div class="lbl">Model Encoding</div></div>',
                unsafe_allow_html=True,
            )
            m4.markdown(
                f'<div class="metric-card"><div class="val">{latency["es_ms"]:.2f} ms</div><div class="lbl">ES Search Execution</div></div>',
                unsafe_allow_html=True,
            )

            st.markdown("---")
            st.markdown(f"### 📋 Search Results ({len(results)})")

            if not results:
                st.info("No matching documents found in index. Try re-indexing the dataset or adjusting your query.")
            else:
                for rank, item in enumerate(results, start=1):
                    snippet = (
                        item["content"][:280] + "..." if len(item["content"]) > 280 else item["content"]
                    )
                    title = item.get("title", f"Document #{item['id']}")

                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                                <div>
                                    <span class="rank-badge">#{rank}</span>
                                    <span class="doc-title">{title}</span>
                                    <span style="font-size: 0.85rem; color: #64748b; margin-left: 0.5rem;">(ID: {item['id']})</span>
                                </div>
                                <span class="score-badge">Score: {item['score']:.4f}</span>
                            </div>
                            <div style="font-size: 0.98rem; color: #334155; line-height: 1.6;">
                                {snippet}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    with st.expander(f"📖 Full Document Context (Doc #{item['id']})"):
                        st.write(item.get("raw_text", item["content"]))


# ------------------------------------------------------------------------------
# TAB 2: Side-by-Side Mode Comparison
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("⚔️ Multi-Mode Comparative Search")
    st.caption("Compare how Lexical BM25, Semantic Dense Vector, and Hybrid Search rank documents for the same query.")

    sample_queries_t2 = [
        "नेपालको इतिहास र संस्कृति",
        "गण्डकी प्रदेश",
        "संवैधानिक गणतन्त्र",
        "भारतीय वाङ्गमय र साहित्य",
    ]

    if "t2_cmp_query" not in st.session_state:
        st.session_state["t2_cmp_query"] = "नेपालको इतिहास र संस्कृति"

    st.markdown("**Quick Preset Queries:**")
    cols_t2 = st.columns(len(sample_queries_t2))

    for idx, q in enumerate(sample_queries_t2):
        if cols_t2[idx].button(q, key=f"ex_t2_{idx}"):
            st.session_state["t2_cmp_query"] = q
            st.rerun()

    cmp_query = st.text_input(
        "Enter test query for comparative evaluation:",
        key="t2_cmp_query",
    )

    if st.button("🚀 Run Comparative Benchmark", key="btn_run_cmp") and cmp_query.strip():
        if not es_connected:
            st.error("Elasticsearch connection offline.")
        else:
            encoder = get_cached_encoder()

            res_hybrid, _, lat_hybrid = perform_search(
                es, encoder, index_name, cmp_query, top_k=top_k, mode="hybrid", bm25_boost=bm25_boost, vector_boost=vector_boost
            )
            res_lexical, _, lat_lexical = perform_search(
                es, encoder, index_name, cmp_query, top_k=top_k, mode="lexical"
            )
            res_vector, _, lat_vector = perform_search(
                es, encoder, index_name, cmp_query, top_k=top_k, mode="vector"
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### ⚡ Hybrid Search")
                st.caption(f"Latency: **{lat_hybrid['total_ms']:.1f} ms**")
                for r_idx, item in enumerate(res_hybrid, start=1):
                    st.markdown(
                        f"""
                        <div class="result-card" style="padding: 0.8rem;">
                            <div style="font-weight: 700; font-size: 0.9rem;">#{r_idx} {item.get('title', item['id'])}</div>
                            <div style="font-size: 0.8rem; color: #1d4ed8;">Score: {item['score']:.4f}</div>
                            <div style="font-size: 0.82rem; color: #475569; margin-top: 0.3rem;">{item['content'][:120]}...</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with col2:
                st.markdown("#### 🔤 Lexical BM25")
                st.caption(f"Latency: **{lat_lexical['total_ms']:.1f} ms**")
                for r_idx, item in enumerate(res_lexical, start=1):
                    st.markdown(
                        f"""
                        <div class="result-card" style="padding: 0.8rem;">
                            <div style="font-weight: 700; font-size: 0.9rem;">#{r_idx} {item.get('title', item['id'])}</div>
                            <div style="font-size: 0.8rem; color: #059669;">Score: {item['score']:.4f}</div>
                            <div style="font-size: 0.82rem; color: #475569; margin-top: 0.3rem;">{item['content'][:120]}...</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with col3:
                st.markdown("#### 🧠 Dense Vector")
                st.caption(f"Latency: **{lat_vector['total_ms']:.1f} ms**")
                for r_idx, item in enumerate(res_vector, start=1):
                    st.markdown(
                        f"""
                        <div class="result-card" style="padding: 0.8rem;">
                            <div style="font-weight: 700; font-size: 0.9rem;">#{r_idx} {item.get('title', item['id'])}</div>
                            <div style="font-size: 0.8rem; color: #7c3aed;">Score: {item['score']:.4f}</div>
                            <div style="font-size: 0.82rem; color: #475569; margin-top: 0.3rem;">{item['content'][:120]}...</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Overlap Analysis
            ids_h = set(x["id"] for x in res_hybrid)
            ids_l = set(x["id"] for x in res_lexical)
            ids_v = set(x["id"] for x in res_vector)

            common_all = ids_h.intersection(ids_l).intersection(ids_v)
            st.markdown("---")
            st.markdown("##### 📊 Top-K Result Overlap Analysis")
            st.write(f"- Documents shared across **all 3 modes**: **{len(common_all)}**")
            st.write(f"- Lexical vs Vector document overlap: **{len(ids_l.intersection(ids_v))}**")


# ------------------------------------------------------------------------------
# TAB 3: Index Health & Diagnostics
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("📊 Elasticsearch Index Diagnostics & Cluster Health")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Cluster Status")
        if es_connected:
            info = es.info()
            st.json(
                {
                    "cluster_name": info.get("cluster_name"),
                    "version": info.get("version", {}).get("number"),
                    "lucene_version": info.get("version", {}).get("lucene_version"),
                    "tagline": info.get("tagline"),
                }
            )
        else:
            st.error("Elasticsearch server is unreachable.")

    with c2:
        st.markdown("##### Index Configuration")
        st.write(f"- **Index Name:** `{index_name}`")
        st.write(f"- **Embedding Model:** `intfloat/multilingual-e5-large` (1024 dimensions)")
        st.write(f"- **Vector Index Type:** Cosine Similarity kNN `dense_vector`")
        st.write(f"- **Lexical Analyzer:** `whitespace` analyzer")
        st.write(f"- **Total Documents:** `{stats.get('count', 0)}`")

    st.markdown("---")
    st.markdown("##### 🛠️ Trigger Dataset Indexing")
    st.write("Stream records from HuggingFace `wikimedia/wikipedia` (Nepali `20231101.ne`), process text through NepStemmer + Stopwords pipeline, generate 1024-dim dense vectors, and bulk ingest into ES.")
    
    t3_sample_size = st.number_input("Documents to index", min_value=100, max_value=5000, value=500, step=100, key="t3_sample")
    if st.button("🚀 Start Indexing Process", key="btn_t3_index"):
        if not es_connected:
            st.error("Cannot index: ES is offline.")
        else:
            prog_bar = st.progress(0.0)
            st_text = st.empty()

            def cb(pct, msg):
                prog_bar.progress(pct)
                st_text.text(msg)

            try:
                enc = get_cached_encoder()
                count, _ = index_dataset(es, enc, index_name=index_name, sample_size=t3_sample_size, progress_callback=cb)
                st.success(f"Successfully indexed {count} documents into index `{index_name}`!")
                st.rerun()
            except Exception as ex:
                st.error(f"Indexing error: {ex}")


# ------------------------------------------------------------------------------
# TAB 4: Devanagari NLP Sandbox
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("🧪 Devanagari NLP Pipeline Inspector Sandbox")
    st.caption("Test how raw Devanagari text transforms through Unicode normalization, symbol stripping, stopword removal, and suffix stemming.")

    test_input = st.text_area(
        "Enter raw Nepali Devanagari text:",
        value="नेपालको इतिहास र संस्कृतिमा विभिन्न कालखण्डमा ठूलो परिवर्तन भएको थियो।",
        height=100,
        key="t4_nlp_input",
    )

    if test_input.strip():
        steps = preprocess_nepali_text(test_input, return_steps=True)

        st.markdown("#### 🔄 Pipeline Execution Breakdown")

        st.markdown(
            f"""
            <div class="step-box">
                <div class="step-title">Step 1: Unicode Normalization (NFKC & ZWJ Removal)</div>
                <div style="font-family: monospace; color: #0284c7;">{steps['nfkc']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="step-box">
                <div class="step-title">Step 2: Devanagari Symbol & Noise Stripping</div>
                <div style="font-family: monospace; color: #0284c7;">{steps['symbols_stripped']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="step-box">
                <div class="step-title">Step 3: Stopword Filtering</div>
                <div><b>Tokens Retained:</b> <span style="font-family: monospace; color: #16a34a;">{" ".join(steps['stopwords_removed'])}</span></div>
                <div style="margin-top: 0.3rem;"><b>Stopwords Removed:</b> <span style="font-family: monospace; color: #dc2626;">{", ".join(steps['removed_stopwords']) if steps['removed_stopwords'] else "None"}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="step-box">
                <div class="step-title">Step 4: NepStemmer Suffix Stripping (Final Search Form)</div>
                <div style="font-family: monospace; color: #7c3aed; font-size: 1.1rem; font-weight: 700;">{steps['final']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
