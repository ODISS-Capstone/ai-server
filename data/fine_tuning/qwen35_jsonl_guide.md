# Qwen3.5 Reasoning JSONL 작성 가이드

이 디렉터리의 `qwen_reasoning_samples.jsonl`은 ODISS 추론 엔진 역할을 Qwen3.5에 학습시키기 위한 supervised fine-tuning 데이터다.

각 줄은 JSON 객체 하나이며, 줄바꿈으로 레코드를 구분한다. 한 레코드 안의 문자열 줄바꿈은 `\n`으로 이스케이프한다.

중요: `<think>`를 assistant 출력에 직접 넣지 않는다. Qwen이 스스로 추론하고 필요한 API tool을 호출하도록 하는 지시는 `system` 메시지에 넣는다.

## 필수 구조

```json
{
  "messages": [
    {
      "role": "system",
      "content": "당신은 ODISS의 추론 엔진 역할을 수행하는 복약 안전 AI입니다. 사용자 질문과 복약 맥락을 읽고 필요한 공공데이터 API tool을 먼저 호출한 뒤, tool 결과를 근거로 안전하고 짧게 답변하세요. 처방 변경, 임의 중단, 임의 병용을 지시하지 말고 필요한 경우 의사·약사 상담을 권하세요."
    },
    {"role": "user", "content": "..."},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "id": "call_001",
          "type": "function",
          "function": {
            "name": "Tool_Check_DUR_Combination_Contraindication",
            "arguments": "{\"item_name\":\"아스피린\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_001",
      "name": "Tool_Check_DUR_Combination_Contraindication",
      "content": "{\"success\":true,\"items\":[...]}"
    },
    {
      "role": "assistant",
      "content": "최종 답변..."
    }
  ],
  "expected_tools": ["Tool_Check_DUR_Combination_Contraindication"],
  "metadata": {
    "intent": "medication_query",
    "source": "seed",
    "risk": "high",
    "api_family": "dur",
    "format": "qwen3.5_system_tool_calling"
  }
}
```

## 작성 순서

1. `system`에는 ODISS reasoning engine 역할, tool 사용 기준, 답변 안전 규칙을 적는다.
2. `user`에는 `[복약 맥락]`, `[OCR 결과]`, `[사용자 질문]` 같은 섹션으로 입력 상황을 정리한다.
3. tool이 필요하면 첫 assistant 메시지의 `content`는 비워 두고 필요한 `tool_calls`를 둔다.
4. 각 tool 호출마다 `role: "tool"` 메시지를 추가한다. `tool_call_id`는 assistant의 `tool_calls[].id`와 반드시 일치해야 한다.
5. 마지막 assistant 메시지에는 사용자에게 보여줄 최종 답변만 쓴다.
6. `expected_tools`에는 호출한 tool 이름을 순서대로 기록한다.
7. `metadata.format`은 `qwen3.5_system_tool_calling`으로 둔다.

## System Prompt 작성 규칙

system prompt는 모델이 스스로 추론하고 API tool을 호출하게 만드는 명령어다. assistant 출력에 내부 추론 과정을 강제로 쓰게 하지 않는다.

권장 system prompt:

```text
당신은 ODISS의 추론 엔진 역할을 수행하는 복약 안전 AI입니다.

역할:
1. 사용자 질문, OCR 결과, 복약 맥락, 환자 정보를 읽고 질문 의도를 판단합니다.
2. 답변에 필요한 사실이 공공데이터 API로 확인 가능하면 먼저 적절한 tool을 호출합니다.
3. tool 결과를 받은 뒤에는 그 결과를 근거로 최종 답변을 작성합니다.
4. tool이 필요 없는 단순 안내는 tool을 호출하지 않고 답변할 수 있습니다.

답변 안전 규칙:
- 처방 변경, 임의 중단, 임의 병용을 지시하지 않습니다.
- 위험 가능성이 있으면 의사 또는 약사 상담을 권합니다.
- 최종 답변은 짧고 고령 사용자도 이해하기 쉬운 한국어로 작성합니다.
```

## 검증 방법

샘플을 수정한 뒤 전체 테스트를 실행한다.

```bash
pytest
```

OpenAI API로 synthetic 데이터를 만들 때:

```bash
OPENAI_API_KEY=... python scripts/generate_reasoning_dataset.py \
  --count 50 \
  --output data/fine_tuning/qwen_reasoning_synthetic.jsonl
```

LoRA 훈련:

```bash
pip install -r requirements-finetune.txt

python scripts/train_qwen_reasoning_lora.py \
  --model Qwen/Qwen3.5-7B-Instruct \
  --train data/fine_tuning/qwen_reasoning_samples.jsonl \
  --output models/qwen-odiss-reasoning-lora \
  --load-in-4bit
```
