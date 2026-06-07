import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

vi.mock("./api/memoryBrowser", () => ({
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
  beforeEach(() => {
    (window as any).__mockWebSockets = [];
  });

  it("renders the assistant as the default screen", () => {
    render(<App />);

    const brand = screen.getByLabelText("오디스");
    expect(brand).toBeInTheDocument();
    expect(brand.querySelector("img")?.getAttribute("src")).toContain("odiss.png");
    expect(screen.getByRole("button", { name: "직접 입력하기" })).toHaveClass("mic-command-button");
    expect(screen.queryByRole("heading", { name: "오디스에게 말씀하세요" })).not.toBeInTheDocument();
    expect(screen.queryByText("연결됨")).not.toBeInTheDocument();
    expect(screen.queryByText("오디스 대기")).not.toBeInTheDocument();
    expect(screen.queryByText("오디스 시작")).not.toBeInTheDocument();
    expect(screen.queryByText(/가운데 마이크/)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "로그 저장" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "로그 내보내기" })).not.toBeInTheDocument();
    expect(screen.queryByText("바로 시작하기")).not.toBeInTheDocument();
    expect(screen.queryByText("약봉투 사진 확인")).not.toBeInTheDocument();
    expect(screen.queryByText("테스터 설정")).not.toBeInTheDocument();
    expect(screen.queryByText("raw JSON")).not.toBeInTheDocument();
    expect(screen.queryByText("세션 피드백")).not.toBeInTheDocument();
  });

  it("opens camera state when the user asks to take a medication photo", async () => {
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "직접 입력" }));
    fireEvent.change(screen.getByLabelText("대화 입력"), {
      target: { value: "약봉투 사진 찍을게" },
    });
    fireEvent.click(screen.getByRole("button", { name: "전송" }));

    expect(await screen.findByRole("heading", { name: "약봉투를 화면에 맞춰주세요" })).toBeInTheDocument();
    expect(screen.getByText("약봉투 사진 찍을게")).toBeInTheDocument();
    expect(screen.getByText("약 이름이 보이면 제가 사진을 읽겠습니다.")).toBeInTheDocument();
  });

  it("does not open the camera for a plain medication package mention", () => {
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "직접 입력" }));
    fireEvent.change(screen.getByLabelText("대화 입력"), {
      target: { value: "나 약봉투 가지고 있어" },
    });
    fireEvent.click(screen.getByRole("button", { name: "전송" }));

    expect(screen.queryByRole("heading", { name: "약봉투를 화면에 맞춰주세요" })).not.toBeInTheDocument();
  });

  it("can close the camera state from the UI", async () => {
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "직접 입력" }));
    fireEvent.change(screen.getByLabelText("대화 입력"), {
      target: { value: "약봉투 사진 찍을게" },
    });
    fireEvent.click(screen.getByRole("button", { name: "전송" }));
    expect(await screen.findByRole("heading", { name: "약봉투를 화면에 맞춰주세요" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "카메라 닫기" }));

    expect(screen.queryByRole("heading", { name: "약봉투를 화면에 맞춰주세요" })).not.toBeInTheDocument();
  });

  it("closes the camera when the server sends the close-camera UI action", async () => {
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "직접 입력" }));
    fireEvent.change(screen.getByLabelText("대화 입력"), {
      target: { value: "약봉투 사진 찍을게" },
    });
    fireEvent.click(screen.getByRole("button", { name: "전송" }));
    expect(await screen.findByRole("heading", { name: "약봉투를 화면에 맞춰주세요" })).toBeInTheDocument();

    const sockets = (window as any).__mockWebSockets;
    const socket = sockets[sockets.length - 1];
    act(() => {
      socket.onmessage?.({
        data: JSON.stringify({
          type: "response",
          response_type: "assistant_control",
          response_text: "네, 사진 확인을 중단할게요.",
          fast_path: "assistant_camera_cancel",
          ui_action: "close_camera",
          requires_tts: true,
        }),
      });
    });

    await waitFor(() => {
      expect(screen.queryByRole("heading", { name: "약봉투를 화면에 맞춰주세요" })).not.toBeInTheDocument();
    });
  });

  it("keeps the memory browser behind admin mode", () => {
    window.history.pushState({}, "", "/?admin=1");
    render(<App />);

    expect(screen.getByRole("button", { name: "로그 내보내기" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "관리자 메모리" }));

    expect(screen.getByText("환자 메모리 브라우저")).toBeInTheDocument();
    expect(screen.getByLabelText("환자명")).toBeInTheDocument();
    window.history.pushState({}, "", "/");
  });
});
