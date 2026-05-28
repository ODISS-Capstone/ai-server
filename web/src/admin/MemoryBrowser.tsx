import { useMemo, useState } from "react";

import {
  getPatientDetail,
  getPatientRecords,
  readMemoryEntry,
  searchPatients,
} from "../api/memoryBrowser";
import type { MemoryRecord, PatientDetail, PatientSummary } from "../api/types";

const CATEGORY_OPTIONS = [
  { value: "ocr_history", label: "OCR" },
  { value: "prescriptions", label: "처방" },
  { value: "medication_log", label: "복용/상담" },
  { value: "dur_linkage", label: "DUR" },
  { value: "health_supplement", label: "건강기능식품" },
  { value: "feedback", label: "피드백" },
  { value: "current_user_profile", label: "현재 프로필" },
  { value: "current_manual", label: "현재 메모" },
  { value: "context_memory", label: "대화 맥락" },
  { value: "prescription_log", label: "복약 요약" },
];

export default function MemoryBrowser() {
  const [nameQuery, setNameQuery] = useState("");
  const [recordQuery, setRecordQuery] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [selectedCategories, setSelectedCategories] = useState<string[]>(
    CATEGORY_OPTIONS.map((item) => item.value),
  );
  const [patients, setPatients] = useState<PatientSummary[]>([]);
  const [selectedPatient, setSelectedPatient] = useState<PatientSummary | null>(null);
  const [patientDetail, setPatientDetail] = useState<PatientDetail | null>(null);
  const [records, setRecords] = useState<MemoryRecord[]>([]);
  const [selectedEntry, setSelectedEntry] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const selectedCategoryLabel = useMemo(
    () =>
      selectedCategories
        .map((value) => CATEGORY_OPTIONS.find((item) => item.value === value)?.label ?? value)
        .join(", "),
    [selectedCategories],
  );

  async function handleSearch(event: React.FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const response = await searchPatients(nameQuery.trim());
      setPatients(response.patients);
      setSelectedPatient(null);
      setPatientDetail(null);
      setRecords([]);
      setSelectedEntry("");
    } catch (searchError) {
      setError(searchError instanceof Error ? searchError.message : "검색에 실패했습니다.");
    } finally {
      setLoading(false);
    }
  }

  async function handleSelectPatient(patient: PatientSummary) {
    setLoading(true);
    setError("");
    setSelectedPatient(patient);
    try {
      const [detail, recordResponse] = await Promise.all([
        getPatientDetail(patient.speaker_id),
        getPatientRecords(patient.speaker_id, {
          categories: selectedCategories,
          query: recordQuery.trim(),
          start: startDate || undefined,
          end: endDate || undefined,
        }),
      ]);
      setPatientDetail(detail);
      setRecords(recordResponse.records);
      setSelectedEntry("");
    } catch (selectError) {
      setError(selectError instanceof Error ? selectError.message : "환자 정보를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }

  async function handleRefreshRecords() {
    if (!selectedPatient) {
      return;
    }
    setLoading(true);
    setError("");
    try {
      const recordResponse = await getPatientRecords(selectedPatient.speaker_id, {
        categories: selectedCategories,
        query: recordQuery.trim(),
        start: startDate || undefined,
        end: endDate || undefined,
      });
      setRecords(recordResponse.records);
    } catch (refreshError) {
      setError(refreshError instanceof Error ? refreshError.message : "기록 검색에 실패했습니다.");
    } finally {
      setLoading(false);
    }
  }

  async function handleOpenRecord(path: string) {
    setLoading(true);
    setError("");
    try {
      const entry = await readMemoryEntry(path);
      setSelectedEntry(entry.content);
    } catch (entryError) {
      setError(entryError instanceof Error ? entryError.message : "원문을 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }

  function toggleCategory(value: string) {
    setSelectedCategories((current) =>
      current.includes(value) ? current.filter((item) => item !== value) : [...current, value],
    );
  }

  return (
    <section className="admin-browser">
      <div className="section-title">
        <div>
          <p className="eyebrow">Admin</p>
          <h2>환자 메모리 브라우저</h2>
        </div>
        <span className="status-pill">read only</span>
      </div>

      <div className="admin-layout">
        <aside className="panel">
          <h3>환자 검색</h3>
          <form className="search-form" onSubmit={handleSearch}>
            <input
              value={nameQuery}
              onChange={(event) => setNameQuery(event.target.value)}
              placeholder="예: 김영수"
              aria-label="환자명"
            />
            <button type="submit" disabled={loading || !nameQuery.trim()}>
              검색
            </button>
          </form>

          {error ? <div className="notice error">{error}</div> : null}

          <div className="patient-list">
            {patients.length === 0 ? (
              <div className="notice muted">검색 결과가 없습니다.</div>
            ) : (
              patients.map((patient) => (
                <button
                  key={patient.speaker_id}
                  type="button"
                  className={`patient-card ${
                    selectedPatient?.speaker_id === patient.speaker_id ? "active" : ""
                  }`}
                  onClick={() => handleSelectPatient(patient)}
                >
                  <strong>{patient.name || patient.speaker_id}</strong>
                  <span>
                    {patient.gender || "-"} · {patient.age ? `${patient.age}세` : "나이 미상"}
                  </span>
                  <small>speaker_id: {patient.speaker_id}</small>
                  {patient.conditions.length ? <small>기저질환: {patient.conditions.join(", ")}</small> : null}
                </button>
              ))
            )}
          </div>
        </aside>

        <main className="panel">
          {!selectedPatient ? (
            <div className="notice muted">왼쪽에서 환자를 검색하고 선택하세요.</div>
          ) : (
            <>
              <h3>{selectedPatient.name || selectedPatient.speaker_id} 상세</h3>
              <div className="profile-grid">
                <div>
                  <strong>성별</strong>
                  <p>{selectedPatient.gender || "-"}</p>
                </div>
                <div>
                  <strong>나이</strong>
                  <p>{selectedPatient.age ? `${selectedPatient.age}세` : "-"}</p>
                </div>
                <div>
                  <strong>최근 사용</strong>
                  <p>{selectedPatient.last_seen_at || "-"}</p>
                </div>
                <div>
                  <strong>확인 시각</strong>
                  <p>{selectedPatient.verified_at || "-"}</p>
                </div>
              </div>

              <div className="filters">
                <input
                  value={recordQuery}
                  onChange={(event) => setRecordQuery(event.target.value)}
                  placeholder="기록 검색어"
                />
                <input type="date" value={startDate} onChange={(event) => setStartDate(event.target.value)} />
                <input type="date" value={endDate} onChange={(event) => setEndDate(event.target.value)} />
                <button type="button" onClick={handleRefreshRecords} disabled={loading}>
                  다시 검색
                </button>
              </div>

              <p className="meta-text">선택 카테고리: {selectedCategoryLabel}</p>
              <div className="category-grid">
                {CATEGORY_OPTIONS.map((option) => (
                  <label key={option.value}>
                    <input
                      type="checkbox"
                      checked={selectedCategories.includes(option.value)}
                      onChange={() => toggleCategory(option.value)}
                    />
                    {option.label}
                  </label>
                ))}
              </div>

              {patientDetail?.history_markdown ? (
                <section className="record-section">
                  <h4>환자 history</h4>
                  <pre className="markdown-view">{patientDetail.history_markdown}</pre>
                </section>
              ) : null}

              <section className="record-section">
                <h4>관련 기록 ({records.length})</h4>
                <div className="record-list">
                  {records.length === 0 ? (
                    <div className="notice muted">선택한 조건과 맞는 기록이 없습니다.</div>
                  ) : (
                    records.map((record) => (
                      <article key={record.path} className="record-item">
                        <header>
                          <div>
                            <strong>{record.category_label}</strong>
                            <span>{record.date}</span>
                          </div>
                          <button type="button" onClick={() => handleOpenRecord(record.path)}>
                            원문 보기
                          </button>
                        </header>
                        <small>{record.path}</small>
                        <p>{record.snippet}</p>
                      </article>
                    ))
                  )}
                </div>
              </section>

              {selectedEntry ? (
                <section className="record-section">
                  <h4>선택 기록 원문</h4>
                  <pre className="markdown-view">{selectedEntry}</pre>
                </section>
              ) : null}
            </>
          )}
        </main>
      </div>
    </section>
  );
}
