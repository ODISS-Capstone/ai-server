# ODISS Custom Scenario Note Example

이 파일을 복사해서 원하는 대화 시나리오로 수정한 뒤 실행합니다.

```bash
./odiss/bin/python scripts/validate_backend_live.py \
  --scenario-file scripts/scenario_note_example.md \
  --scenario my_prescription_followup \
  --strict
```

아래 JSON 블록만 harness가 읽습니다. Markdown 설명은 자유롭게 추가해도 됩니다.

```json
{
  "scenarios": [
    {
      "id": "my_prescription_followup",
      "speaker_id": "note_prescription_followup",
      "runner": "orchestrator",
      "seed_medications": ["타이레놀정", "이부프로펜정", "알마겔정"],
      "steps": [
        {
          "id": "interaction_check",
          "text": "이 약들 같이 먹어도 되나요?",
          "expected_mode": "tool_first",
          "expected_intent": "medication_query",
          "expected_terms": ["타이레놀", "이부프로펜"]
        },
        {
          "id": "stomach_med_recall",
          "text": "그중 위장약은 뭐예요?",
          "expected_mode": "tool_first",
          "expected_intent": "medication_query",
          "expected_terms": ["알마겔"]
        },
        {
          "id": "previous_context_recall",
          "text": "아까 말한 약 다시 쉽게 설명해줘",
          "expected_mode": "memory_only",
          "expected_terms": ["타이레놀", "이부프로펜", "알마겔"]
        }
      ]
    },
    {
      "id": "my_websocket_dialogue",
      "speaker_id": "note_websocket_dialogue",
      "runner": "websocket",
      "seed_medications": ["아스피린정"],
      "steps": [
        {
          "id": "greeting",
          "text": "안녕하세요.",
          "expected_mode": "memory_only",
          "expected_intent": "smalltalk"
        },
        {
          "id": "medicine_question",
          "text": "아스피린정 먹을 때 주의할 점 알려줘",
          "expected_mode": "tool_first",
          "expected_intent": "medication_query",
          "expected_terms": ["아스피린"]
        }
      ]
    }
  ]
}
```

## 필드 설명

- `id`: 리포트와 `--scenario`에서 사용할 시나리오 이름
- `speaker_id`: 같은 화자 메모리로 묶을 기준 이름
- `runner`: `orchestrator` 또는 `websocket`
- `seed_medications`: 테스트 시작 전에 OCR/DUR seed로 저장할 약 이름
- `steps[].text`: 사용자 발화
- `steps[].expected_mode`: `tool_first`, `memory_only`, `frontier_first`, `ask_user_clarify`
- `steps[].expected_intent`: 예: `smalltalk`, `medication_query`, `emergency`, `unknown`
- `steps[].expected_terms`: 최종 응답, 핵심 메시지, recall source 중 반드시 포함되어야 하는 단어
- `steps[].require_disclaimer`: 기본값 `true`, clarify/smalltalk에서 안전 문구를 요구하지 않으려면 `false`
