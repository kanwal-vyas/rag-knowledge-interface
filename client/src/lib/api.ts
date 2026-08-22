// Signal Room style reminder: API helpers stay invisible and precise; the UI names only states the backend can support.

export type Latency = {
  total_ms?: number | null;
  stt_ms?: number | null;
  sparse_retrieval_ms?: number | null;
  dense_retrieval_ms?: number | null;
  context_assembly_ms?: number | null;
  generation_ms?: number | null;
  guardrail_ms?: number | null;
  error?: string;
};

export type QueryResponse = {
  answer: string;
  grounded: boolean;
  request_id: string;
  latency: Latency;
};

export type HealthResponse = {
  status: string;
  environment: string;
  providers: Record<string, string>;
};

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

async function parseResponse<T>(response: Response): Promise<T> {
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new Error("The knowledge service returned an unreadable response.");
  }

  if (!response.ok) {
    const detail = typeof payload === "object" && payload !== null && "detail" in payload
      ? String((payload as { detail?: unknown }).detail || "Request was not accepted.")
      : "Request was not accepted.";
    throw new Error(detail);
  }
  return payload as T;
}

function detectLanguage(text: string): string {
  // Detect if the query contains Devanagari (Hindi) characters
  const devanagariChars = text.match(/[\u0900-\u097F]/g);
  if (devanagariChars && devanagariChars.length > text.replace(/\s/g, '').length * 0.3) {
    return "hi";
  }
  return "en";
}

export async function queryText(query_text: string): Promise<QueryResponse> {
  const lang = detectLanguage(query_text);
  const response = await fetch(`${API_BASE_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query_text, lang, retrieval_mode: "cross_lingual", top_k: 10 }),
  });
  return parseResponse<QueryResponse>(response);
}

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`);
  return parseResponse<HealthResponse>(response);
}
