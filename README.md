# OCR 기반 멀티모달 시니어 복약지도 서버

인식 → 추론 → 전송 파이프라인으로, 처방전/약봉투 이미지 OCR, DUR 검증, 내부·외부 LLM 추론, MCP 연동, 기기 전송까지 구현한 서버입니다.

## 구조

- **인식:** 입력(음성/이미지) → OCR(DeepSeek) → DUR(KPIC) → DB/문서화
- **추론:** 내부 LLM + 개인정보 검열 → 외부 모델 강화 → 팩트 검증 → MCP 전달
- **전송:** MCP 데이터 통합 → 의사소통 AI 에이전트 → 서버 API → 기기 음성 출력

## 실행

```bash
pip install -r requirements.txt
cp .env.example .env   # API 키 등 설정
python -m uvicorn app.main:app --reload --host 0.0.0.0
```

- API 문서: http://localhost:8000/docs
- Health: http://localhost:8000/health
- 파이프라인: POST /query/pipeline (이미지 업로드 → OCR → DUR → DB/문서화)
- 답변 생성: POST /query/ask (session_id, query_text 선택) → 내부 LLM → 검열 → 외부 LLM → 검증 → 시니어 친화 → MCP/기기 전송

## 환경 변수

`.env.example` 참고. 필수: `DATABASE_URL`, OCR/DUR/LLM 관련 API URL·키(해당 단계 사용 시).
