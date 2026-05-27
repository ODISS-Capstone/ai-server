# ODISS AI 복약 비서 웹앱

배포용 웹 테스트 앱입니다. 일반 사용자는 `/app`에서 대화, 브라우저 STT/TTS, OCR 이미지 업로드를 사용할 수 있고, 관리자는 메모리 브라우저와 raw payload를 확인할 수 있습니다.

## 실행

```bash
cp .env.example .env.local
npm install
npm run dev
```

개발 서버: <http://localhost:5173/app/>

FastAPI 단일 서버 배포 시:

```bash
npm run build
```

빌드 결과물 `web/dist`는 FastAPI가 `/app`으로 정적 서빙합니다.

## 환경 변수

| 변수 | 설명 |
|------|------|
| `VITE_API_BASE_URL` | API 베이스 URL. 비우면 현재 origin과 Vite proxy를 사용합니다. |
| `VITE_MEMORY_BROWSER_TOKEN` | 선택 사항. 관리자 메모리 브라우저용 fallback 토큰입니다. 기본은 런타임 입력 토큰을 사용합니다. |

토큰은 빌드에 박지 않고 화면 입력 또는 `?token=` 초대 링크로 받습니다.

## 주요 기능

- WebSocket `/ws/chat` 기반 대화
- 브라우저 STT, 텍스트 fallback, 브라우저 TTS
- filler 발화 on/off, TTS 중지
- 모바일 카메라/이미지 업로드 후 `/upload/image` OCR 실행
- OCR 결과를 `ocr_result` 메시지로 WebSocket 대화 흐름에 연결
- 관리자 모드에서 raw JSON, latency, fast path, 메모리 브라우저 확인
