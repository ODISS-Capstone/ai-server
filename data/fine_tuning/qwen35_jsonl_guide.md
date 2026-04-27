# Qwen3.5 Reasoning JSONL 작성 가이드

이 디렉터리의 `qwen_reasoning_samples.jsonl`은 ODISS 추론 엔진 역할을 Qwen3.5에 학습시키기 위한 supervised fine-tuning 데이터다.

각 줄은 JSON 객체 하나이며, 줄바꿈으로 레코드를 구분한다. 한 레코드 안의 문자열 줄바꿈은 `\n`으로 이스케이프한다.

## 필수 구조

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {
      "role": "assistant",
      "content": "<think>\nintent: ...\nneeded_tools: ...\nsafety_policy: ...\n</think>",
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
      "content": "<think>\ntool_result_summary: ...\nanswer_policy: ...\n</think>\n최종 답변..."
    }
  ],
  "expected_tools": ["Tool_Check_DUR_Combination_Contraindication"],
  "metadata": {
    "intent": "medication_query",
    "source": "seed",
    "risk": "high",
    "api_family": "dur",
    "format": "qwen3.5_think_tool_calling"
  }
}
```

## 작성 순서

1. `system`에는 ODISS reasoning engine 역할과 Qwen3.5 `<think>` 형식 요구를 적는다.
2. `user`에는 `[복약 맥락]`, `[OCR 결과]`, `[사용자 질문]` 같은 섹션으로 입력 상황을 정리한다.
3. 첫 assistant 메시지에는 `<think>` 블록을 넣고, 이어서 필요한 `tool_calls`를 둔다.
4. 각 tool 호출마다 `role: "tool"` 메시지를 추가한다. `tool_call_id`는 assistant의 `tool_calls[].id`와 반드시 일치해야 한다.
5. 마지막 assistant 메시지에는 tool 결과 요약 `<think>` 블록과 사용자에게 보여줄 최종 답변을 함께 쓴다.
6. `expected_tools`에는 호출한 tool 이름을 순서대로 기록한다.
7. `metadata.format`은 `qwen3.5_think_tool_calling`으로 둔다.

## `<think>` 블록 규칙

`<think>`는 모델 학습용 구조화 추론 요약이다. 장문 사유 과정이나 불필요한 내면 독백을 넣지 않는다.

Tool 호출 전:

```text
<think>
intent: medication_query
needed_tools: 병용 금기와 노인주의 확인
safety_policy: 출혈 위험 가능성이 있으므로 임의 병용 허용 답변을 피한다.
</think>
```

Tool 결과 후:

```text
<think>
tool_result_summary: 병용 금기 결과에서 출혈 위험 증가 가능성이 확인됨.
answer_policy: 안전성 단정 금지, 처방자 또는 약사 확인 권고.
</think>
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
