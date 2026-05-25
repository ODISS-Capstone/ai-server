import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import App from "./App";

vi.mock("../api/memoryBrowser", () => ({
  searchPatients: vi.fn().mockResolvedValue({
    query: "김영수",
    total: 1,
    patients: [
      {
        speaker_id: "patient-kim",
        name: "김영수",
        age: "72",
        gender: "남성",
        conditions: ["고혈압"],
        last_seen_at: "2026-05-19T12:00:00",
        verified_at: "2026-05-19T12:00:00",
      },
    ],
  }),
  getPatientDetail: vi.fn(),
  getPatientRecords: vi.fn(),
  readMemoryEntry: vi.fn(),
}));

describe("App", () => {
  it("renders patient search UI", () => {
    render(<App />);
    expect(screen.getByText("ODISS 환자 메모리 검색")).toBeInTheDocument();
    expect(screen.getByLabelText("환자명")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "검색" })).toBeInTheDocument();
  });
});
