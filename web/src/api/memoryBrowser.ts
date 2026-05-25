import type {
  MemoryEntryResponse,
  PatientDetail,
  PatientRecordsResponse,
  PatientSearchResponse,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") ?? "";
const TOKEN = import.meta.env.VITE_MEMORY_BROWSER_TOKEN ?? "";

function authHeaders(): HeadersInit {
  const headers: HeadersInit = { Accept: "application/json" };
  if (TOKEN) {
    headers.Authorization = `Bearer ${TOKEN}`;
  }
  return headers;
}

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { headers: authHeaders() });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function searchPatients(name: string, limit = 20): Promise<PatientSearchResponse> {
  const params = new URLSearchParams({ name, limit: String(limit) });
  return request<PatientSearchResponse>(`/api/memory/patients?${params}`);
}

export function getPatientDetail(speakerId: string): Promise<PatientDetail> {
  return request<PatientDetail>(`/api/memory/patients/${encodeURIComponent(speakerId)}`);
}

export function getPatientRecords(
  speakerId: string,
  options?: {
    categories?: string[];
    query?: string;
    start?: string;
    end?: string;
    limit?: number;
  },
): Promise<PatientRecordsResponse> {
  const params = new URLSearchParams();
  if (options?.categories?.length) {
    params.set("categories", options.categories.join(","));
  }
  if (options?.query) {
    params.set("query", options.query);
  }
  if (options?.start) {
    params.set("start", options.start);
  }
  if (options?.end) {
    params.set("end", options.end);
  }
  if (options?.limit) {
    params.set("limit", String(options.limit));
  }
  const query = params.toString();
  const suffix = query ? `?${query}` : "";
  return request<PatientRecordsResponse>(
    `/api/memory/patients/${encodeURIComponent(speakerId)}/records${suffix}`,
  );
}

export function readMemoryEntry(path: string): Promise<MemoryEntryResponse> {
  const params = new URLSearchParams({ path });
  return request<MemoryEntryResponse>(`/api/memory/entry?${params}`);
}
