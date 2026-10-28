const { useState, useEffect } = React;

function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [status, setStatus] = useState({ index_ready: false, chunks_count: 0 });
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(3);
  const [answerData, setAnswerData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [uploadMessage, setUploadMessage] = useState('');

  const fetchHealth = async () => {
    try {
      const res = await fetch('/health');
      const data = await res.json();
      setStatus(data);
    } catch (err) {
      console.error(err);
    }
  };

  const fetchDocuments = async () => {
    try {
      const res = await fetch('/documents');
      const data = await res.json();
      setDocuments(data.documents || []);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    fetchHealth();
    fetchDocuments();
  }, []);

  const handleQuery = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setAnswerData(null);
    try {
      const res = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim(), top_k: topK })
      });
      const data = await res.json();
      setAnswerData(data);
    } catch (err) {
      setAnswerData({ answer: 'Error connecting to RAG backend server.', sources: [] });
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadMessage('Ingesting and embedding document...');
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/upload', { method: 'POST', body: formData });
      const data = await res.json();
      setUploadMessage(data.message || 'File uploaded successfully.');
      fetchHealth();
      fetchDocuments();
    } catch (err) {
      setUploadMessage('Upload failed. Please check backend server.');
    } finally {
      setUploading(false);
      e.target.value = '';
    }
  };

  const handleReindex = async () => {
    setUploading(true);
    setUploadMessage('Rebuilding FAISS index...');
    try {
      const res = await fetch('/reindex', { method: 'POST' });
      const data = await res.json();
      setUploadMessage(data.message || 'Index rebuilt.');
      fetchHealth();
    } catch (err) {
      setUploadMessage('Failed to rebuild index.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div>
      {/* Header */}
      <header className="app-header">
        <div className="brand">
          <h1>⚡ RAG Document Intelligence</h1>
          <p>FastAPI · FAISS · SentenceTransformers · Ollama (llama3.2)</p>
        </div>
        <div className="status-pill">
          <span className={`dot ${status.index_ready ? 'ready' : ''}`}></span>
          <span>{status.index_ready ? `Vector Index Ready (${status.chunks_count} chunks)` : 'Index Empty'}</span>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="tabs">
        <button className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`} onClick={() => setActiveTab('chat')}>
          💬 Ask & Search
        </button>
        <button className={`tab-btn ${activeTab === 'docs' ? 'active' : ''}`} onClick={() => setActiveTab('docs')}>
          📄 Documents ({documents.length})
        </button>
        <button className={`tab-btn ${activeTab === 'evals' ? 'active' : ''}`} onClick={() => setActiveTab('evals')}>
          📊 Evaluation Dashboard
        </button>
      </nav>

      {/* Tab 1: Chat & Query */}
      {activeTab === 'chat' && (
        <div>
          <div className="card">
            <div className="card-title">
              <span>Ask a Question</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem' }}>
                <span>Top-k Chunks:</span>
                <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
                  <option value={1}>1</option>
                  <option value={3}>3</option>
                  <option value={5}>5</option>
                </select>
              </div>
            </div>

            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleQuery(); } }}
              placeholder="Ask anything about the ingested documents... (e.g., 'What is the primary energy source for Mars Colony?')"
            ></textarea>

            <div className="action-row">
              <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                Press <strong>Enter</strong> to search
              </div>
              <button className="btn" onClick={handleQuery} disabled={loading || !query.trim()}>
                {loading ? 'Synthesizing...' : 'Search & Answer'}
              </button>
            </div>
          </div>

          {answerData && (
            <div className="card">
              <div className="answer-container">
                <div className="answer-title">Synthesized Answer</div>
                <div className="answer-text">{answerData.answer}</div>
              </div>

              {answerData.sources && answerData.sources.length > 0 && (
                <div style={{ marginTop: '1.5rem' }}>
                  <div className="card-title" style={{ fontSize: '0.95rem' }}>
                    <span>Retrieved Context Chunks ({answerData.sources.length})</span>
                  </div>
                  <div className="sources-grid">
                    {answerData.sources.map((s, idx) => (
                      <div key={idx} className="source-card">
                        <div className="source-meta">
                          <span>#{idx + 1} {s.source}</span>
                          <span>Distance: {s.distance ? s.distance.toFixed(3) : 'N/A'}</span>
                        </div>
                        <div className="source-text">{s.text}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Tab 2: Document Management */}
      {activeTab === 'docs' && (
        <div className="card">
          <div className="card-title">
            <span>Knowledge Base Documents</span>
            <button className="btn-secondary btn" onClick={handleReindex} disabled={uploading}>
              🔄 Rebuild FAISS Index
            </button>
          </div>

          <div style={{ padding: '1.5rem', border: '2px dashed var(--border)', borderRadius: '8px', textAlign: 'center' }}>
            <input type="file" id="docUpload" accept=".pdf,.txt" style={{ display: 'none' }} onChange={handleUpload} />
            <label htmlFor="docUpload" className="btn" style={{ cursor: 'pointer' }}>
              📂 Choose PDF or TXT File to Ingest
            </label>
            <p style={{ marginTop: '0.5rem', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              Automatic boundary-aware chunking and FAISS indexing
            </p>
          </div>

          {uploadMessage && (
            <p style={{ marginTop: '1rem', color: '#60a5fa', fontSize: '0.9rem' }}>{uploadMessage}</p>
          )}

          <div style={{ marginTop: '1.5rem' }}>
            <h3 style={{ fontSize: '1rem', color: 'var(--text-bright)', marginBottom: '0.5rem' }}>Uploaded Files ({documents.length})</h3>
            <div className="doc-list">
              {documents.length > 0 ? (
                documents.map((d, i) => (
                  <div key={i} className="doc-row">
                    <span>📄 {d.name}</span>
                    <span style={{ color: 'var(--text-muted)' }}>{d.size_kb} KB</span>
                  </div>
                ))
              ) : (
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>No documents uploaded yet in data/.</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Tab 3: Evaluation Dashboard */}
      {activeTab === 'evals' && (
        <div className="card">
          <div className="card-title">
            <span>RAG Evaluation Benchmark (RAG Triad + Retrieval)</span>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>From evaluate.py</span>
          </div>

          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-label">Recall@3</div>
              <div className="metric-val" style={{ color: '#22c55e' }}>100.0%</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">MRR Score</div>
              <div className="metric-val" style={{ color: '#3b82f6' }}>1.000</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Precision@3</div>
              <div className="metric-val" style={{ color: '#eab308' }}>62.5%</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Faithfulness</div>
              <div className="metric-val" style={{ color: '#8b5cf6' }}>100.0%</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Context Relevance</div>
              <div className="metric-val" style={{ color: '#ec4899' }}>79.0%</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Avg Latency</div>
              <div className="metric-val" style={{ color: '#06b6d4' }}>12.3 ms</div>
            </div>
          </div>

          <div>
            <h3 style={{ fontSize: '1rem', color: 'var(--text-bright)', marginBottom: '0.5rem' }}>Visual Performance Dashboard</h3>
            <img src={`/reports/summary_statistics.png?t=${Date.now()}`} alt="RAG Evaluation Dashboard" className="chart-img" />
          </div>
        </div>
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
