# Qwen3.5 Reasoning JSONL 작성 가이드

이 디렉터리의 `qwen_reasoning_samples.jsonl`은 ODISS 추론 엔진 역할을 Qwen3.5에 학습시키기 위한 supervised fine-tuning 데이터다.

각 줄은 JSON 객체 하나이며, 줄바꿈으로 레코드를 구분한다. 한 레코드 안의 문자열 줄바꿈은 `\n`으로 이스케이프한다.

중요: 런타임 프롬프트와 훈련용 데이터는 다르다.

- 런타임: `app/prompts/qwen35_reasoning_template.md`의 system prompt가 Qwen에 제공된다. 이때 assistant 출력에 `<think>`를 요구하지 않는다.
- 훈련 데이터: Qwen이 추론 엔진 역할을 학습하도록 assistant 메시지에 `<think>...</think>` longCOT 블록을 포함한다.
- 훈련 데이터의 system 메시지에는 런타임과 동일하게 tool 사용 기준과 tool call API 양식을 넣는다.

## 필수 구조

```json
{
  "messages": [
    {
      "role": "system",
      "content": "당신은 ODISS의 추론 엔진 역할을 수행하는 복약 안전 AI입니다. 사용자 질문과 복약 맥락을 읽고 필요한 공공데이터 API tool을 먼저 호출한 뒤, tool 결과를 근거로 안전하고 짧게 답변하세요. Tool call API 양식: tool 호출이 필요하면 assistant 메시지는 content를 빈 문자열로 두고 tool_calls 배열을 포함합니다. 각 tool call은 id, type: \"function\", function.name, function.arguments를 포함합니다. function.arguments는 JSON 문자열입니다. 서버가 tool 실행 결과를 반환하면 role: \"tool\", tool_call_id, name, content 형식의 메시지로 받습니다. 처방 변경, 임의 중단, 임의 병용을 지시하지 말고 필요한 경우 의사·약사 상담을 권하세요."
    },
    {"role": "user", "content": "..."},
    {
      "role": "assistant",
      "content": "<think>\n1. 의도: medication_query\n2. 근거: 복수 약물 병용 질문이며 와파린 복용 중이다.\n3. 필요한 확인: 병용 금기 및 고령자 주의 여부를 DUR API로 확인한다.\n4. 안전 방침: 병용 가능 여부를 단정하지 않고 tool 결과 확인 후 상담 권고로 답한다.\n</think>",
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
      "content": "<think>\n1. tool 결과 요약: 병용 시 출혈 위험 증가와 고령자 출혈 주의가 확인되었다.\n2. 답변 결정: 임의 병용을 피하고 처방자/약사에게 와파린 복용 사실을 알리도록 안내한다.\n3. 안전 문구: 최종 판단은 전문가 상담이 필요함을 포함한다.\n</think>\n최종 답변..."
    }
  ],
  "expected_tools": ["Tool_Check_DUR_Combination_Contraindication"],
  "metadata": {
    "intent": "medication_query",
    "source": "seed",
    "risk": "high",
    "api_family": "dur",
    "format": "qwen3.5_longcot_tool_calling"
  }
}
```

## 작성 순서

1. `system`에는 ODISS reasoning engine 역할, tool 사용 기준, tool call API 양식, 답변 안전 규칙을 적는다.
2. `user`에는 `[복약 맥락]`, `[OCR 결과]`, `[사용자 질문]` 같은 섹션으로 입력 상황을 정리한다.
3. tool이 필요하면 첫 assistant 메시지의 `content`에 `<think>...</think>` longCOT 블록을 쓰고 필요한 `tool_calls`를 둔다.
4. 각 tool 호출마다 `role: "tool"` 메시지를 추가한다. `tool_call_id`는 assistant의 `tool_calls[].id`와 반드시 일치해야 한다.
5. 마지막 assistant 메시지에는 tool 결과를 해석하는 `<think>...</think>` longCOT 블록과 사용자에게 보여줄 최종 답변을 함께 쓴다.
6. `expected_tools`에는 호출한 tool 이름을 순서대로 기록한다.
7. `metadata.format`은 `qwen3.5_longcot_tool_calling`으로 둔다.

## System Prompt 작성 규칙

system prompt는 모델이 스스로 추론하고 API tool을 호출하게 만드는 명령어다. OpenAI-compatible tool call API 양식도 system prompt 안에 명시한다. 다만 훈련 데이터에서는 assistant가 어떤 판단 흐름으로 tool을 고르는지 학습하도록 별도의 `<think>...</think>` longCOT 블록을 assistant 메시지에 포함한다.

권장 system prompt:

```text
당신은 ODISS의 추론 엔진 역할을 수행하는 복약 안전 AI입니다.

역할:
1. 사용자 질문, OCR 결과, 복약 맥락, 환자 정보를 읽고 질문 의도를 판단합니다.
2. 답변에 필요한 사실이 공공데이터 API로 확인 가능하면 먼저 적절한 tool을 호출합니다.
3. tool 결과를 받은 뒤에는 그 결과를 근거로 최종 답변을 작성합니다.
4. tool이 필요 없는 단순 안내는 tool을 호출하지 않고 답변할 수 있습니다.

Tool call API 양식:
- tool 호출이 필요하면 assistant 메시지는 content를 빈 문자열로 둡니다.
- assistant 메시지에는 tool_calls 배열을 포함합니다.
- 각 tool call은 id, type: "function", function.name, function.arguments를 포함합니다.
- function.arguments는 JSON 문자열이며 tool schema에 정의된 인자만 포함합니다.
- tool 결과는 role: "tool", tool_call_id, name, content 형식으로 받습니다.

답변 안전 규칙:
- 처방 변경, 임의 중단, 임의 병용을 지시하지 않습니다.
- 위험 가능성이 있으면 의사 또는 약사 상담을 권합니다.
- 최종 답변은 짧고 고령 사용자도 이해하기 쉬운 한국어로 작성합니다.
```

## LongCOT 작성 규칙

훈련용 `<think>` 블록은 모델이 reasoning engine 역할을 학습하도록 충분히 자세히 쓴다. 단, 개인정보나 실제 환자 식별 정보는 포함하지 않는다.

Tool 호출 전 assistant:

```text
<think>
1. 의도: medication_query
2. 입력 근거: 사용자가 두 약의 병용 가능 여부를 묻고 있으며, 와파린 복용 중이라는 고위험 맥락이 있다.
3. 위험 후보: 항응고제와 진통소염제 병용 시 출혈 위험이 증가할 수 있고, 고령자에서는 이상반응 위험이 더 커질 수 있다.
4. 필요한 API: 병용 금기 여부는 Tool_Check_DUR_Combination_Contraindication, 고령자 주의는 Tool_Check_DUR_Geriatric_Caution으로 확인한다.
5. tool 인자 결정: 사용자 입력에 명시된 새 처방약 이름인 아스피린 장용정을 item_name으로 사용한다.
6. 답변 보류: API 결과 확인 전에는 복용 가능 여부를 단정하지 않는다.
</think>
```

Tool 결과 후 최종 assistant:

```text
<think>
1. tool 결과 요약: 병용 시 출혈 위험 증가 가능성과 고령자 출혈 주의가 확인되었다.
2. 근거 평가: 두 tool 결과가 같은 방향의 안전 우려를 제시하므로 병용을 안전하다고 안내하면 안 된다.
3. 답변 전략: 사용자가 이해하기 쉬운 문장으로 위험 가능성을 설명하고, 임의 복용 대신 처방자 또는 약사에게 확인하도록 안내한다.
4. 안전 문구: 정확한 판단은 의사·약사 상담이 필요하다는 문구를 포함한다.
</think>
```

## 검증 방법

샘플을 수정한 뒤 전체 테스트를 실행한다.

```bash
pytest
```

## OpenAI API로 synthetic 데이터 만들기 (in-context learning)

스크립트는 `data/fine_tuning/qwen_reasoning_samples.jsonl`의 손작성 샘플을
in-context exemplar로 자동 주입한다. GPT는 이 예시들의 JSON 모양, longCOT
구조, 한국어 톤을 따라 **새로운** 시나리오를 만든다.

### 1. API 키 입력 위치

다음 중 한 곳만 채우면 된다(우선순위 높은 순):

1. `--api-key sk-...` 명령행 인자
2. `OPENAI_API_KEY` 환경변수
3. `ai-server/.env`의 `OPENAI_API_KEY=sk-...` 줄 (`.env.example` 참고)

`.env`에 있는 값은 스크립트가 시작 시 자동으로 로드한다.

```bash
# ai-server/.env 에 한 줄 추가하면 충분
echo 'OPENAI_API_KEY=sk-...' >> ai-server/.env
# 모델 변경이 필요하면
echo 'OPENAI_DATASET_MODEL=gpt-5.5' >> ai-server/.env
```

### 2. 100개 샘플 생성 레시피

```bash
cd ai-server

# 먼저 prompt가 의도대로 만들어지는지 dry-run으로 확인
python scripts/generate_reasoning_dataset.py --dry-run --exemplars 2

# 실제 100개 생성 (intent 5종을 라운드로빈으로 순환)
python scripts/generate_reasoning_dataset.py \
  --count 100 \
  --exemplars 2 \
  --output data/fine_tuning/qwen_reasoning_synthetic.jsonl
```

시나리오 `intent`는 `SCENARIO_SEEDS`에 정의된 5종(medication_query,
drug_identification, supplement_query, duration_or_dosage,
pregnancy_or_age_specific)으로 라운드로빈된다. count=100이면 각 intent당
20샘플이 만들어진다.

각 호출마다 GPT는 다음 입력을 받는다.

* `task` 명령
* `in_context_examples`: 같은 intent 우선으로 시드에서 1–3개를 골라 넣음
* `required_output_schema`: 출력 JSON 스키마
* `constraints`: 톤·안전·tool 제약 등
* `scenario_seed`: 이번 샘플의 intent/환자/질문 좌표
* `available_tools`: `app/prompts/llm_tools.json`에서 추출한 사용 가능 tool

응답은 `response_format=json_object`로 강제하고,
`scripts/generate_reasoning_dataset.py:validate_sample`이
구조·`<think>`·tool 이름 유효성을 즉시 검사한다. 검증 실패 샘플은 건너뛴다.

### 3. 비용·속도 가이드

* 평균 한 샘플 = prompt 약 4–6 KB + 응답 약 1–2 KB.
* 기본 모델은 `gpt-5.5`다. 가격은 변동될 수 있으므로 실행 전 OpenAI 계정의 최신 usage/pricing 화면에서 확인한다.
* 100샘플은 직렬 호출 기준 몇 분에서 수십 분까지 걸릴 수 있다. 더 빠르게 만들고 싶으면 `--exemplars 1`로 줄이거나 `--model`로 저지연 모델을 명시한다.
* GPT-5 계열(`gpt-5`, `gpt-5.5` 등)은 Chat Completions에서 `temperature`를 받지 않으므로 스크립트가 자동으로 생략한다. `--temperature`는 GPT-4 계열 같은 레거시 모델에만 적용된다.
* 결과 파일은 `--output`으로 지정하며, **append 모드**로 열린다. 실패 후
  재시도 시 같은 파일에 이어 붙으니, 새 파일을 원하면 다른 경로를 주거나
  먼저 지운다.

### 4. 옵션 표

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--count` | 20 | 생성할 샘플 수 |
| `--model` | `gpt-5.5` (env: `OPENAI_DATASET_MODEL`) | OpenAI 모델 id |
| `--exemplars` | 2 | 한 호출에 in-context로 넣을 시드 샘플 수(0이면 비활성) |
| `--exemplars-path` | `data/fine_tuning/qwen_reasoning_samples.jsonl` | exemplar 풀 |
| `--temperature` | 0.7 | GPT-4 계열 등 temperature 지원 모델에서만 사용. GPT-5 계열은 자동 생략 |
| `--seed` | 20260506 | exemplar 셔플 시드(재현 가능) |
| `--max-failures` | 10 | 누적 실패 허용 횟수, 초과 시 abort |
| `--api-key` | (env / .env) | 명령행 직접 지정 시 환경변수 무시 |
| `--dry-run` | off | 첫 prompt만 출력하고 OpenAI 호출 안 함 |

## 엔진 역할 분리 데이터셋 (권장 기본)

현재 런타임 오케스트레이션은 엔진별 계약으로 분리되어 있으므로, 학습 데이터도
`router / memory / delivery` 3종으로 나누는 것을 기본 경로로 사용한다.

### 1) 모놀리식 reasoning 데이터 분해

```bash
cd ai-server
python scripts/split_reasoning_dataset.py
```

기본 출력:

- `data/fine_tuning/qwen_router_samples.jsonl`
- `data/fine_tuning/qwen_memory_samples.jsonl`
- `data/fine_tuning/qwen_delivery_samples.jsonl`

### 2) 아키텍처 적합성 게이트

```bash
python scripts/evaluate_engine_datasets.py \
  --dataset data/fine_tuning/qwen_router_samples.jsonl \
  --task-family router --strict

python scripts/evaluate_engine_datasets.py \
  --dataset data/fine_tuning/qwen_memory_samples.jsonl \
  --task-family memory --strict

python scripts/evaluate_engine_datasets.py \
  --dataset data/fine_tuning/qwen_delivery_samples.jsonl \
  --task-family delivery --strict
```

레거시 `qwen_reasoning_samples.jsonl`은 `<think>` + monolithic tool-calling 학습
검증 용도로만 유지하고, 런타임 직접 배포 모델 학습에는 사용하지 않는 것을 권장한다.

## LoRA 훈련

```bash
pip install -r requirements-finetune.txt

python scripts/train_qwen_reasoning_lora.py \
  --model Qwen/Qwen3.5-7B-Instruct \
  --train data/fine_tuning/qwen_reasoning_samples.jsonl \
  --output models/qwen-odiss-reasoning-lora \
  --load-in-4bit
```

엔진별 헤드는 `--task-family`로 바로 지정할 수 있다:

```bash
python scripts/train_qwen_reasoning_lora.py \
  --task-family router \
  --output models/qwen-odiss-router-lora \
  --load-in-4bit

python scripts/train_qwen_reasoning_lora.py \
  --task-family memory \
  --output models/qwen-odiss-memory-lora \
  --load-in-4bit

python scripts/train_qwen_reasoning_lora.py \
  --task-family delivery \
  --output models/qwen-odiss-delivery-lora \
  --load-in-4bit
```

훈련용 데이터는 시드(`qwen_reasoning_samples.jsonl`)와 합성
(`qwen_reasoning_synthetic.jsonl`) 두 파일을 모두 넘기는 것이 권장된다:

```bash
python scripts/train_qwen_reasoning_lora.py \
  --model Qwen/Qwen3.5-7B-Instruct \
  --train data/fine_tuning/qwen_reasoning_samples.jsonl \
          data/fine_tuning/qwen_reasoning_synthetic.jsonl \
  --output models/qwen-odiss-reasoning-lora \
  --load-in-4bit
```
