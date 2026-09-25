document.addEventListener('DOMContentLoaded', () => {
    // --- UI State & Elements ---
    const elements = {
        tabs: document.querySelectorAll('.tab-btn'),
        panes: document.querySelectorAll('.tab-pane'),
        
        // Sidebar
        statusDot: document.querySelector('.status-indicator .dot'),
        statusText: document.querySelector('.status-indicator .status-text'),
        indexedCount: document.getElementById('indexed-docs-count'),
        topK: document.getElementById('top_k'),
        topKVal: document.getElementById('top_k_val'),
        bm25Boost: document.getElementById('bm25_boost'),
        bm25BoostVal: document.getElementById('bm25_boost_val'),
        vectorBoost: document.getElementById('vector_boost'),
        vectorBoostVal: document.getElementById('vector_boost_val'),
        incrementalFile: document.getElementById('incremental-file'),
        btnUploadIndex: document.getElementById('btn-upload-index'),
        uploadStatus: document.getElementById('upload-status'),
        
        // Search Tab
        searchInput: document.getElementById('search-input'),
        searchMode: document.getElementById('search-mode'),
        btnSearch: document.getElementById('btn-search'),
        searchResults: document.getElementById('search-results-container'),
        presetBtns: document.querySelectorAll('.preset-btn'),
        
        // Compare Tab
        compareInput: document.getElementById('compare-input'),
        btnCompare: document.getElementById('btn-compare'),
        compareContainer: document.getElementById('compare-results-container'),
        
        // Diagnostics Tab
        diagCluster: document.getElementById('diag-cluster-info'),
        diagIndex: document.getElementById('diag-index-info'),
        
        // Sandbox Tab
        sandboxInput: document.getElementById('sandbox-input'),
        btnSandbox: document.getElementById('btn-sandbox'),
        sandboxResults: document.getElementById('sandbox-results')
    };

    // --- Tab Switching ---
    elements.tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            elements.tabs.forEach(t => t.classList.remove('active'));
            elements.panes.forEach(p => p.classList.remove('active'));
            
            tab.classList.add('active');
            document.getElementById(tab.dataset.target).classList.add('active');
        });
    });

    // --- Sidebar Sliders ---
    elements.topK.addEventListener('input', (e) => elements.topKVal.textContent = e.target.value);
    elements.bm25Boost.addEventListener('input', (e) => elements.bm25BoostVal.textContent = e.target.value);
    elements.vectorBoost.addEventListener('input', (e) => elements.vectorBoostVal.textContent = e.target.value);

    // --- Diagnostics (Polling) ---
    async function fetchDiagnostics() {
        try {
            const res = await fetch('/api/diagnostics');
            const data = await res.json();
            
            if (data.es_connected) {
                elements.statusDot.className = 'dot online';
                elements.statusText.textContent = 'ES Connected';
                elements.indexedCount.textContent = data.index_stats?.count || 0;
                
                if (data.cluster_info) {
                    elements.diagCluster.textContent = JSON.stringify(data.cluster_info, null, 2);
                }
                if (data.index_stats) {
                    elements.diagIndex.innerHTML = `
                        <li><span class="diag-label">Index Name</span> <span class="diag-value">${data.index_name}</span></li>
                        <li><span class="diag-label">Status</span> <span class="diag-value">${data.index_stats.status}</span></li>
                        <li><span class="diag-label">Total Documents</span> <span class="diag-value">${data.index_stats.count}</span></li>
                    `;
                }
            } else {
                elements.statusDot.className = 'dot offline';
                elements.statusText.textContent = 'ES Offline';
                elements.diagCluster.textContent = 'Elasticsearch server is unreachable.';
                elements.diagIndex.innerHTML = '<li>Error connecting to ES</li>';
            }
        } catch (e) {
            console.error('Failed to fetch diagnostics:', e);
        }
    }
    
    // Initial fetch and set interval
    fetchDiagnostics();
    setInterval(fetchDiagnostics, 10000);

    // --- Incremental Indexing ---
    elements.btnUploadIndex.addEventListener('click', async () => {
        const file = elements.incrementalFile.files[0];
        if (!file) {
            elements.uploadStatus.textContent = "Please select a CSV file.";
            return;
        }
        
        elements.uploadStatus.textContent = "Uploading and indexing...";
        elements.btnUploadIndex.disabled = true;
        
        const formData = new FormData();
        formData.append("file", file);
        
        try {
            const res = await fetch('/api/upload_index', {
                method: 'POST',
                body: formData
            });
            
            const data = await res.json();
            
            if (!res.ok) {
                throw new Error(data.error || `HTTP ${res.status}`);
            }
            
            elements.uploadStatus.textContent = `✅ Added ${data.indexed_count}; skipped ${data.duplicate_count} duplicates.`;
            elements.incrementalFile.value = ""; // clear input
            fetchDiagnostics();
        } catch (e) {
            elements.uploadStatus.textContent = `❌ Failed: ${e.message}`;
        } finally {
            elements.btnUploadIndex.disabled = false;
        }
    });

    // --- Search Logic ---
    elements.presetBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            elements.searchInput.value = btn.dataset.query;
            elements.btnSearch.click();
        });
    });

    function createTelemetryBar(lat) {
        return `
            <div class="telemetry-bar">
                <div class="tel-item"><div class="tel-val">${lat.total_ms.toFixed(1)} ms</div><div class="tel-lbl">Total Latency</div></div>
                <div class="tel-item"><div class="tel-val">${lat.prep_ms.toFixed(1)} ms</div><div class="tel-lbl">NLP Preprocessing</div></div>
                <div class="tel-item"><div class="tel-val">${lat.encoding_ms.toFixed(1)} ms</div><div class="tel-lbl">Model Encoding</div></div>
                <div class="tel-item"><div class="tel-val">${lat.es_ms.toFixed(1)} ms</div><div class="tel-lbl">ES Search</div></div>
            </div>
        `;
    }

    elements.btnSearch.addEventListener('click', async () => {
        const query = elements.searchInput.value;
        if (!query.trim()) return;

        elements.searchResults.innerHTML = '<div class="status-message">Searching...</div>';
        elements.btnSearch.disabled = true;

        try {
            const res = await fetch('/api/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: query,
                    mode: elements.searchMode.value,
                    top_k: elements.topK.value,
                    bm25_boost: elements.bm25Boost.value,
                    vector_boost: elements.vectorBoost.value
                })
            });
            const data = await res.json();
            
            if (data.error) {
                elements.searchResults.innerHTML = `<div class="status-message" style="color:red">Error: ${data.error}</div>`;
                return;
            }

            let html = `
                <div class="prep-inspector">
                    <div>Raw Query: <strong>${query}</strong></div>
                    <div>Cleaned & Stemmed: <div class="prep-box">${data.cleaned_query || '(empty)'}</div></div>
                </div>
            `;
            html += createTelemetryBar(data.latency);

            if (data.results.length === 0) {
                html += '<div class="status-message">No matching documents found.</div>';
            } else {
                const tpl = document.getElementById('tpl-result-card').content;
                const fragment = document.createDocumentFragment();
                
                data.results.forEach((item, index) => {
                    const clone = document.importNode(tpl, true);
                    clone.querySelector('.rank-badge').textContent = `#${index + 1}`;
                    clone.querySelector('.doc-title').textContent = item.title;
                    clone.querySelector('.doc-id').textContent = `(ID: ${item.id})`;
                    clone.querySelector('.score-badge').textContent = `Score: ${item.score.toFixed(4)}`;
                    
                    let snippet = item.content.length > 280 ? item.content.substring(0, 280) + '...' : item.content;
                    clone.querySelector('.result-snippet').textContent = snippet;
                    clone.querySelector('.full-doc-content').textContent = item.raw_text;
                    
                    fragment.appendChild(clone);
                });
                
                const resultsWrapper = document.createElement('div');
                resultsWrapper.appendChild(fragment);
                html += resultsWrapper.innerHTML;
            }
            
            elements.searchResults.innerHTML = html;
        } catch (e) {
            elements.searchResults.innerHTML = `<div class="status-message" style="color:red">Request failed.</div>`;
        } finally {
            elements.btnSearch.disabled = false;
        }
    });

    // --- Compare Logic ---
    elements.btnCompare.addEventListener('click', async () => {
        const query = elements.compareInput.value;
        if (!query.trim()) return;

        elements.btnCompare.disabled = true;
        elements.compareContainer.style.display = 'grid';
        
        ['hybrid', 'lexical', 'vector'].forEach(mode => {
            document.getElementById(`comp-res-${mode}`).innerHTML = 'Loading...';
        });

        try {
            const res = await fetch('/api/compare', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: query,
                    top_k: elements.topK.value,
                    bm25_boost: elements.bm25Boost.value,
                    vector_boost: elements.vectorBoost.value
                })
            });
            const data = await res.json();
            
            if (data.error) throw new Error(data.error);

            ['hybrid', 'lexical', 'vector'].forEach(mode => {
                const modeData = data[mode];
                document.getElementById(`comp-lat-${mode}`).textContent = `Latency: ${modeData.latency.total_ms.toFixed(1)} ms`;
                
                const container = document.getElementById(`comp-res-${mode}`);
                container.innerHTML = '';
                
                modeData.results.forEach((item, idx) => {
                    let color = mode === 'hybrid' ? '#1d4ed8' : (mode === 'lexical' ? '#059669' : '#7c3aed');
                    container.innerHTML += `
                        <div class="comp-card">
                            <div class="comp-card-header">
                                <span>#${idx+1} ${item.title}</span>
                                <span style="color:${color}">Score: ${item.score.toFixed(4)}</span>
                            </div>
                            <div style="color: var(--text-muted);">${item.content.substring(0, 120)}...</div>
                        </div>
                    `;
                });
            });

        } catch (e) {
            alert('Comparison failed: ' + e.message);
        } finally {
            elements.btnCompare.disabled = false;
        }
    });

    // --- Sandbox Logic ---
    elements.btnSandbox.addEventListener('click', async () => {
        const text = elements.sandboxInput.value;
        if (!text.trim()) return;

        elements.btnSandbox.disabled = true;
        elements.sandboxResults.style.display = 'block';
        elements.sandboxResults.innerHTML = 'Processing...';

        try {
            const res = await fetch('/api/nlp_sandbox', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });
            const data = await res.json();
            const s = data.steps;

            elements.sandboxResults.innerHTML = `
                <div class="step-box">
                    <div class="step-title">Step 1: Unicode Normalization</div>
                    <div class="step-content">${s.nfkc}</div>
                </div>
                <div class="step-box">
                    <div class="step-title">Step 2: Symbol Stripping</div>
                    <div class="step-content">${s.symbols_stripped}</div>
                </div>
                <div class="step-box">
                    <div class="step-title">Step 3: Stopword Filtering</div>
                    <div class="step-content">Tokens: ${s.stopwords_removed.join(' ')}</div>
                    <div class="step-content" style="color: var(--danger); font-size: 0.85rem; margin-top: 0.2rem">Removed: ${s.removed_stopwords.join(', ')}</div>
                </div>
                <div class="step-box">
                    <div class="step-title">Step 4: NepStemmer Output (Final)</div>
                    <div class="step-content highlight">${s.final}</div>
                </div>
            `;
        } catch (e) {
            elements.sandboxResults.innerHTML = 'Error running sandbox pipeline.';
        } finally {
            elements.btnSandbox.disabled = false;
        }
    });
});
