import { useEffect, useRef, useState } from "react";

type IngestResponse = {
  file_id: string;
  duplicate: boolean;
  doc_name?: string;
};

type QueryResponse = {
  answer?: string;
  decision?: string;
  complexity?: string;
  trace?: Array<Record<string, unknown>>;
  time_ms?: number;
  cache_status?: string;
  token_usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
};

export default function App() {
  const [fileId, setFileId] = useState<string | null>(null);
  const [docName, setDocName] = useState<string | null>(null);
  const [duplicate, setDuplicate] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string>("No file uploaded");
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [trace, setTrace] = useState<QueryResponse["trace"]>([]);
  const [decision, setDecision] = useState<string | undefined>();
  const [complexity, setComplexity] = useState<string | undefined>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cacheStatus, setCacheStatus] = useState<string | null>(null);
  const [cacheIndicator, setCacheIndicator] = useState<string>("—");
  const [retrievalTimeMs, setRetrievalTimeMs] = useState<number | null>(null);
  const [tokenUsage, setTokenUsage] = useState<QueryResponse["token_usage"] | null>(null);

  const apiBase = "http://127.0.0.1:8000";
  const withBase = (path: string) => (apiBase ? `${apiBase}${path}` : path);
  const endpoint = withBase("/query");
  const ingestEndpoint = withBase("/ingest");
  const clearCacheEndpoint = withBase("/cache/clear");
  const cacheCheckEndpoint = withBase("/cache/check");

  const cacheTimer = useRef<number | null>(null);

  const formatCacheStatus = (status: string) => {
    switch (status) {
      case "hit_exact":
        return "Cache hit (exact)";
      case "hit_semantic":
        return "Cache hit (semantic)";
      case "hit_retrieval":
        return "Cache hit (retrieval)";
      case "miss":
        return "Cache miss";
      default:
        return "—";
    }
  };

  const formatFallback = (value: unknown) => {
    if (value === undefined || value === null) return "—";
    if (typeof value === "boolean") return value ? "Yes" : "No";
    return String(value);
  };

  const getFinalFallback = (entries: QueryResponse["trace"]) => {
    if (!entries || entries.length === 0) return undefined;
    for (let i = entries.length - 1; i >= 0; i -= 1) {
      const entry = entries[i] as { data?: Record<string, unknown> } | undefined;
      const data = entry?.data;
      if (data && Object.prototype.hasOwnProperty.call(data, "fallback")) {
        return (data as { fallback?: unknown }).fallback;
      }
    }
    return undefined;
  };

  const getDocsFound = (entries: QueryResponse["trace"]) => {
    if (!entries || entries.length === 0) return undefined;
    for (const entry of entries) {
      const data = (entry as { data?: Record<string, unknown> } | undefined)?.data;
      if (data && Object.prototype.hasOwnProperty.call(data, "docs_found")) {
        return (data as { docs_found?: unknown }).docs_found;
      }
    }
    return undefined;
  };

  const finalFallback = getFinalFallback(trace);
  const docsFound = getDocsFound(trace);
  const totalTokens = tokenUsage?.total_tokens;

  const displayDocName = (name: string | null) => {
    if (!name) return "No document loaded yet";
    const underscoreIndex = name.indexOf("_");
    if (underscoreIndex > -1 && underscoreIndex < name.length - 1) {
      return name.slice(underscoreIndex + 1);
    }
    return name;
  };

  const onUpload = async (file: File) => {
    setError(null);
    setUploadStatus("Uploading...");
    setBusy(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(ingestEndpoint, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || "Upload failed");
      }
      const data = (await res.json()) as IngestResponse;
      setFileId(data.file_id);
      setDocName(data.doc_name ?? null);
      setDuplicate(data.duplicate);
      setUploadStatus(
        data.duplicate ? "Duplicate file detected" : "File ingested"
      );
    } catch (err) {
      setError((err as Error).message);
      setUploadStatus("Upload failed");
    } finally {
      setBusy(false);
    }
  };

  const onRunQuery = async () => {
    setError(null);
    setCacheStatus(null);
    setBusy(true);
    setAnswer("");
    setTrace([]);
    setDecision(undefined);
    setComplexity(undefined);
    setRetrievalTimeMs(null);
    setTokenUsage(null);
    try {
      if (!fileId) {
        throw new Error("Please upload a file before running a query.");
      }
      if (!query.trim()) {
        throw new Error("Please enter a question.");
      }
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, file_id: fileId }),
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || "Query failed");
      }
      const data = (await res.json()) as QueryResponse;
      setAnswer(data.answer ?? "");
      setTrace(data.trace ?? []);
      setDecision(data.decision);
      setComplexity(data.complexity);
      setRetrievalTimeMs(data.time_ms ?? null);
      setTokenUsage(data.token_usage ?? null);
      if (data.cache_status) {
        setCacheIndicator(data.cache_status);
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const onClearCache = async () => {
    setError(null);
    setCacheStatus(null);
    setBusy(true);
    try {
      const res = await fetch(clearCacheEndpoint, { method: "POST" });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || "Clear cache failed");
      }
      const data = (await res.json()) as {
        cleared?: { exact?: number; semantic?: number; retrieval?: number };
      };
      const exact = data.cleared?.exact ?? 0;
      const semantic = data.cleared?.semantic ?? 0;
      const retrieval = data.cleared?.retrieval ?? 0;
      setCacheStatus(
        `Cache cleared (exact: ${exact}, semantic: ${semantic}, retrieval: ${retrieval})`
      );
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    if (cacheTimer.current) {
      window.clearTimeout(cacheTimer.current);
    }
    if (!fileId || !query.trim()) {
      setCacheIndicator("—");
      return;
    }
    cacheTimer.current = window.setTimeout(async () => {
      try {
        const res = await fetch(cacheCheckEndpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, file_id: fileId }),
        });
        if (!res.ok) {
          setCacheIndicator("—");
          return;
        }
        const data = (await res.json()) as {
          status?: string;
          time_ms?: number;
          similarity?: number;
        };
        setCacheIndicator(formatCacheStatus(data.status ?? "—"));
      } catch {
        setCacheIndicator("—");
      }
    }, 400);
    return () => {
      if (cacheTimer.current) {
        window.clearTimeout(cacheTimer.current);
      }
    };
  }, [cacheCheckEndpoint, fileId, query]);

  return (
    <div className="app">
      <header className="hero">
        <div>
          <p className="eyebrow">Agentic RAG Workspace</p>
          <h1>Agentic AI</h1>
          <p className="subtitle">
            Upload once, ask anything. The system routes your query and traces
            every decision.
          </p>
        </div>
      </header>

      <section className="top-row">
        <div className="pill-card">
          <div className="pill-title">Upload status</div>
          <div className="pill-value">{displayDocName(docName)}</div>
        </div>
        <div className="pill-card">
          <div className="pill-title">Duplicate check</div>
          <div className="pill-value">{duplicate ? "Duplicate" : "Unique"}</div>
          <div className="pill-sub">
            {duplicate
              ? "Using existing embeddings"
              : "Ready to build fresh index"}
          </div>
        </div>
        <div className="pill-card">
          <div className="pill-title">Model route</div>
          <div className="pill-value">{complexity ?? "—"}</div>
          <div className="pill-sub">A: simple, B: retrieval, C: multi-hop</div>
        </div>
      </section>

      <section className="grid">
        <aside className="side">
          <div className="panel">
            <h3>Controls</h3>
            <label className="upload">
              <input
                type="file"
                accept=".json,.pdf,.docx,.pptx,.txt"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) onUpload(file);
                }}
                disabled={busy}
              />
              Upload file
            </label>
            <div className="actions">
              <button className="secondary" onClick={onClearCache} disabled={busy}>
                Clear cache
              </button>
            </div>
            {error ? <div className="error">{error}</div> : null}
            {cacheStatus ? <div className="muted">{cacheStatus}</div> : null}
          </div>
          <div className="panel">
            <h3>Run diagnostics</h3>
            <div className="diag-row">
              <span>Decision</span>
              <strong>{decision ?? "—"}</strong>
            </div>
            <div className="diag-row">
              <span>Cache</span>
              <strong>{cacheIndicator}</strong>
            </div>
            <div className="diag-row">
              <span>Retrieval time</span>
              <strong>{retrievalTimeMs != null ? `${retrievalTimeMs} ms` : "—"}</strong>
            </div>
            <div className="diag-row">
              <span>Fallback used</span>
              <strong>{formatFallback(finalFallback)}</strong>
            </div>
            <div className="diag-row">
              <span>Docs found</span>
              <strong>{docsFound != null ? String(docsFound) : "—"}</strong>
            </div>
            <div className="diag-row">
              <span>Tokens used</span>
              <strong>{totalTokens != null ? String(totalTokens) : "—"}</strong>
            </div>
            <div className="diag-row">
              <span>Trace events</span>
              <strong>{trace?.length ?? 0}</strong>
            </div>
          </div>
        </aside>

        <div className="panel">
          <div className="panel-header">
            <div>
              <h2>Ask a question</h2>
              <p>Enter a question, then run the agentic flow.</p>
            </div>
          </div>
          <textarea
            className="textarea"
            placeholder="Type your query here..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <div className="actions">
            <button className="primary" onClick={onRunQuery} disabled={busy}>
              {busy ? "Running..." : "Run query"}
            </button>
          </div>
          <div className="answer">
            <h3>Final answer</h3>
            <p>{answer || "Run a query to see the answer."}</p>
          </div>
        </div>
      </section>
    </div>
  );
}

