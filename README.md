# OCR 기반 멀티모달 복약관리 서버

인식 → 추론 → 전송 파이프라인으로, 처방전/약봉투 이미지 OCR, DUR 검증, 내부·외부 LLM 추론, MCP 연동, 기기 전송까지 구현한 서버입니다.

## 구조

- **인식:** 입력(음성/이미지) → OCR(DeepSeek) → DUR(KPIC) → DB/문서화
- **추론:** 내부 LLM + 개인정보 검열 → 외부 모델 강화 → 팩트 검증 → MCP 전달
- **전송:** MCP 데이터 통합 → 의사소통 AI 에이전트 → 서버 API → 기기 음성 출력
- **메모리:** 기존 `flash/permanent` MD 저장소 + Claude Code 스타일 `structured_memory`

## 구조화 메모리

서버는 `data/md_database/structured_memory` 아래에 전역 메모리와 화자별 메모리를 분리해 저장합니다.

- `global/MEMORY.md`: 팀 공통 인덱스
- `global/*.md`: 매뉴얼, 공통 레퍼런스
- `speakers/{speaker_id}/MEMORY.md`: 화자별 인덱스
- `speakers/{speaker_id}/*.md`: 환자 프로필, 최근 복약 맥락, DUR 관련 장기 메모

이 계층은 다음 흐름으로 동작합니다.

- frontmatter 기반 topic file 저장
- `MEMORY.md` 인덱스 자동 재생성
- 얕은 스캔으로 메모리 헤더만 빠르게 수집
- 질의와 실제로 매칭되는 메모리만 선별
- 오래된 메모리에는 freshness 경고 추가

## 실행

```bash
pip install -r requirements.txt
cp .env.example .env   # API 키 등 설정
python -m uvicorn app.main:app --reload --host 0.0.0.0
```

- API 문서: http://localhost:8000/docs
- Health: http://localhost:8000/health
- 파이프라인: POST /query/pipeline (이미지 업로드 → OCR → DUR → DB/문서화)
- 답변 생성: POST /query/ask (session_id, query_text 선택) → 내부 LLM → 검열 → 외부 LLM → 검증 → 사용자 친화 → MCP/기기 전송

## Qwen3-4B + TurboQuant 상시 검증

ai-server는 내부 LLM을 직접 추론하지 않고 OpenAI-compatible LLM 서버를 `INTERNAL_LLM_API_URL`로 호출한다. 따라서 Qwen3-4B는 별도 모델 서버 프로세스로 계속 띄우고, ai-server는 그 endpoint를 바라보게 한다.

> 주의: `app.services.turboquant_runtime`의 auto-wrap은 해당 Python 프로세스 안에서 `transformers`로 로드되는 모델에 적용된다. 별도 LLM 서버를 띄울 때도 그 서버 프로세스에 `TurboQuantWrapper`를 설치하고 `TURBOQUANT_*` 환경 변수를 함께 넘겨야 한다.

### 1. 모델 파일 배치

서버에 모델 디렉터리를 만든다.

```bash
mkdir -p /home/jepetolee/models
```

Hugging Face에서 받는 경우:

```bash
huggingface-cli download Qwen/Qwen3-4B \
  --local-dir /home/jepetolee/models/qwen3-4b
```

이미 받은 모델을 서버로 업로드하는 경우:

```bash
scp -r ./Qwen3-4B user@server:/home/jepetolee/models/qwen3-4b
```

### 2. TurboQuantWrapper 설치

TurboQuantWrapper 저장소를 같은 서버에 두고 editable로 설치한다.

```bash
git clone <TurboQuantWrapper repo url> /home/jepetolee/TurboQuantWrapper
python3 -m pip install --upgrade pip
python3 -m pip install "transformers>=4.44.0" "accelerate>=0.30.0" sentencepiece
python3 -m pip install -e /home/jepetolee/TurboQuantWrapper
```

설치 확인:

```bash
python3 - <<'PY'
import turboquant.runtime as tq
print("TurboQuant runtime OK:", tq)
PY
```

Qwen3-4B wrapper smoke가 필요하면 TurboQuantWrapper의 테스트 스크립트를 먼저 돌린다.

```bash
python3 /home/jepetolee/TurboQuantWrapper/scripts/test_turboquant_wrapping.py \
  --targets qwen3-4b \
  --report turboquant-wrap-qwen3-4b.json
```

### 3. LLM 서버 상시 실행

운영 중 계속 떠 있어야 하므로 `tmux`에서 실행한다.

```bash
tmux new -s qwen3-llm
```

LLM 서버 프로세스에는 TurboQuant 환경 변수를 같이 준다.

```bash
export CUDA_VISIBLE_DEVICES=0
export TURBOQUANT_AUTO_WRAP=true
export TURBOQUANT_KEY_BITS=3
export TURBOQUANT_VALUE_BITS=3
export TURBOQUANT_COMPRESS_VALUES=false
export TURBOQUANT_REQUIRE_CUDA=true

python3 -m vllm.entrypoints.openai.api_server \
  --model /home/jepetolee/models/qwen3-4b \
  --served-model-name qwen3-4b \
  --host 0.0.0.0 \
  --port 8001 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --trust-remote-code
```

`Ctrl+B`, `D`로 tmux에서 빠져나오면 서버는 계속 실행된다. 다시 들어가려면:

```bash
tmux attach -t qwen3-llm
```

### 4. ai-server 연결

ai-server의 `.env` 또는 실행 환경에 내부 LLM endpoint를 지정한다.

```bash
INTERNAL_LLM_API_URL=http://localhost:8001/v1/chat/completions
INTERNAL_LLM_API_KEY=
INTERNAL_LLM_MODEL=qwen3-4b

LOG_LEVEL=INFO
LOG_TO_FILE=true
LOG_FILE_PATH=./logs/ai-server.log
```

ai-server 실행:

```bash
python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. 검증

LLM 서버를 직접 확인한다.

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-4b",
    "messages": [{"role": "user", "content": "/no_think\nping"}],
    "max_tokens": 32
  }'
```

ai-server 경유 확인:

```bash
curl http://localhost:8000/health/llm
```

로그 확인:

```bash
tail -f logs/ai-server.log
```

정상 연결이면 `logs/ai-server.log`에 `[InternalLLMHealth] check_ok`가 남는다. GPU 적재 상태는 별도 터미널에서 확인한다.

```bash
watch -n 1 nvidia-smi
```

## 테스트

```bash
pytest
```

`pytest.ini`에서 `tests/`만 수집하도록 고정해 두었기 때문에, 루트의 실험용 스크립트는 자동 수집되지 않습니다.

### 엔진 계약/데이터셋 게이트

`ai-server/.github/workflows/ci.yml`의 `test` job은 아래 계약 게이트를 추가로 강제한다.

```bash
python scripts/split_reasoning_dataset.py

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

레거시 monolithic 셋(`qwen_reasoning_samples.jsonl`)은 아래 명령으로 런타임 불일치(`"<think>"`, tool-calling 혼합)를 리포트한다.

```bash
python scripts/evaluate_engine_datasets.py \
  --dataset data/fine_tuning/qwen_reasoning_samples.jsonl \
  --task-family reasoning
```

## 환경 변수

`.env.example` 참고. 필수: `DATABASE_URL`, OCR/DUR/LLM 관련 API URL·키(해당 단계 사용 시).
