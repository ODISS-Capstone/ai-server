# ODISS 환자 메모리 브라우저

환자명 검색 → 프로필·기록 열람 (read-only).

## 실행

1. AI 서버 `.env`에 `MEMORY_BROWSER_TOKEN` 설정 (development에서 비워두면 토큰 없이 접근 가능)
2. AI 서버 기동 (`:8000`)
3. 프론트:

```bash
cp .env.example .env.local
npm install
npm run dev
```

http://localhost:5173

## 환경 변수 (`.env.local`)

| 변수 | 설명 |
|------|------|
| `VITE_API_BASE_URL` | API 베이스 (dev에서는 Vite proxy 사용, 기본 `http://127.0.0.1:8000`) |
| `VITE_MEMORY_BROWSER_TOKEN` | 서버 `MEMORY_BROWSER_TOKEN`과 동일 |

## API (프록시 `/api/memory`)

- `GET /api/memory/patients?name=` — 환자 검색
- `GET /api/memory/patients/{speaker_id}` — 프로필 상세
- `GET /api/memory/patients/{speaker_id}/records` — 기록 검색
- `GET /api/memory/entry?path=` — Markdown 원문
