import { useEffect, useMemo, useState } from "react";

import MemoryBrowser from "./admin/MemoryBrowser";
import AssistantApp from "./assistant/AssistantApp";
import { saveToken, storedToken, tokenFromUrl } from "./api/assistant";

type Tab = "assistant" | "admin";

export default function App() {
  const params = useMemo(() => new URLSearchParams(window.location.search), []);
  const initialAdmin = params.get("admin") === "1";
  const initialToken = tokenFromUrl() || storedToken();
  const [token, setToken] = useState(initialToken);
  const [tokenInput, setTokenInput] = useState(initialToken);
  const [adminMode, setAdminMode] = useState(initialAdmin);
  const [tab, setTab] = useState<Tab>("assistant");

  useEffect(() => {
    saveToken(token);
  }, [token]);

  function handleTokenSave() {
    setToken(tokenInput.trim());
  }

  return (
    <div className={`app-shell ${adminMode ? "admin-enabled" : ""}`}>
      <header className="app-topbar">
        <div className="brand-mark" aria-label="오디스">
          <img src={`${import.meta.env.BASE_URL}odiss.png`} alt="" />
        </div>
        {adminMode ? (
          <details className="tester-settings">
            <summary>관리자 설정</summary>
            <div className="header-controls">
              <label className="token-box">
                접속 토큰
                <input
                  value={tokenInput}
                  onChange={(event) => setTokenInput(event.target.value)}
                  placeholder="초대 링크 토큰 또는 직접 입력"
                />
              </label>
              <button type="button" onClick={handleTokenSave}>
                저장
              </button>
              <label className="toggle-row">
                <input
                  type="checkbox"
                  checked={adminMode}
                  onChange={(event) => {
                    setAdminMode(event.target.checked);
                    if (!event.target.checked) {
                      setTab("assistant");
                    }
                  }}
                />
                관리자 모드
              </label>
            </div>
          </details>
        ) : null}
      </header>

      {adminMode ? (
        <nav className="tab-bar" aria-label="화면 전환">
          <button
            type="button"
            className={tab === "assistant" ? "active" : ""}
            onClick={() => setTab("assistant")}
          >
            복약 비서
          </button>
          <button type="button" className={tab === "admin" ? "active" : ""} onClick={() => setTab("admin")}>
            관리자 메모리
          </button>
        </nav>
      ) : null}

      {tab === "assistant" ? <AssistantApp token={token} adminMode={adminMode} /> : <MemoryBrowser />}
    </div>
  );
}
