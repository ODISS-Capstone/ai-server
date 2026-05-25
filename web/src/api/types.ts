export interface PatientSummary {
  speaker_id: string;
  name: string;
  age: string;
  gender: string;
  conditions: string[];
  last_seen_at: string;
  verified_at: string;
}

export interface PatientSearchResponse {
  query: string;
  patients: PatientSummary[];
  total: number;
}

export interface PatientDetail {
  speaker_id: string;
  profile: Record<string, unknown>;
  profile_markdown: string;
  history_markdown: string;
  medication_events_markdown: string;
  structured_memory: {
    memory_index: string;
    memory_prompt: string;
    relevant_memories: Array<Record<string, unknown>>;
  };
}

export interface MemoryRecord {
  category: string;
  category_label: string;
  date: string;
  path: string;
  snippet: string;
  preview: string;
}

export interface PatientRecordsResponse {
  speaker_id: string;
  profile: PatientSummary;
  keywords: string[];
  records: MemoryRecord[];
  total: number;
}

export interface MemoryEntryResponse {
  path: string;
  content: string;
}
