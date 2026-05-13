좋습니다. 지금 적어준 건 **3가지가 아니라 4가지**로 보는 게 맞습니다.

1. 다중화자 / 신규 유저 등록
2. 장기간 일반 대화 + 짧은 스몰토킹
3. 날짜가 지나도 메모리가 유지되는지 확인
4. OCR 약물 인식 + DUR API + 복용지도 + 스몰토킹

아래처럼 노트를 만들면 됩니다.
파일명 예시는 `scripts/my_odiss_scenarios.md`로 두면 됩니다.

````md
# ODISS 내가 원하는 커스텀 시나리오

아래 JSON 블록만 validate_backend_live.py가 읽습니다.

```json
{
  "scenarios": [
    {
      "id": "multi_speaker_register_male_user",
      "speaker_id": "multi_user_minsu_32_male",
      "runner": "orchestrator",
      "seed_medications": [],
      "steps": [
        {
          "id": "new_user_introduces_profile",
          "text": "처음 왔어요. 제 이름은 김민수고 32살 남자예요. 앞으로 제 약 복용을 도와줬으면 좋겠어요.",
          "expected_mode": "memory_only",
          "expected_intent": "smalltalk",
          "expected_terms": ["김민수"],
          "require_disclaimer": false
        },
        {
          "id": "profile_recall",
          "text": "내 이름이랑 나이 기억해?",
          "expected_mode": "memory_only",
          "expected_intent": "smalltalk",
          "expected_terms": ["김민수", "32"],
          "require_disclaimer": false
        }
      ]
    },
    {
      "id": "multi_speaker_register_female_user",
      "speaker_id": "multi_user_jiyoon_67_female",
      "runner": "orchestrator",
      "seed_medications": [],
      "steps": [
        {
          "id": "new_user_introduces_profile",
          "text": "안녕하세요. 저는 박지윤이고 67살 여자예요. 약 이름을 자주 헷갈려서 도움을 받고 싶어요.",
          "expected_mode": "memory_only",
          "expected_intent": "smalltalk",
          "expected_terms": ["박지윤"],
          "require_disclaimer": false
        },
        {
          "id": "profile_recall",
          "text": "내 이름이랑 나이 기억하고 있어?",
          "expected_mode": "memory_only",
          "expected_intent": "smalltalk",
          "expected_terms": ["박지윤", "67"],
          "require_disclaimer": false
        }
      ]
    },
    {
      "id": "long_term_smalltalk_general_conversation",
      "speaker_id": "long_term_smalltalk_user_001",
      "runner": "orchestrator",
      "seed_medications": [],
      "steps": [
        {
          "id": "day_life_log",
          "text": "오늘 아침에 산책을 30분 했고, 점심은 죽을 먹었어. 요즘은 약보다도 생활습관을 잘 챙기고 싶어.",
          "expected_mode": "memory_only",
          "expected_intent": "smalltalk",
          "expected_terms": ["산책", "30분"],
          "require_disclaimer": false
        },
        {
          "id": "short_smalltalk",
          "text": "오늘은 그냥 짧게 안부만 물어봐줘.",
          "expected_mode": "memory_only",
          "expected_intent": "smalltalk",
          "expected_terms": [],
          "require_disclaimer": false
        },
        {
          "id": "recall_lifestyle_context",
          "text": "아까 내가 생활습관 관련해서 뭐 했다고 했지?",
          "expected_mode": "memory_only",
          "expected_intent": "smalltalk",
          "expected_terms": ["산책", "30분"],
          "require_disclaimer": false
        },
        {
          "id": "gentle_followup_smalltalk",
          "text": "그럼 내일도 부담 없이 할 수 있는 한 가지를 추천해줘.",
          "expected_mode": "memory_only",
          "expected_intent": "smalltalk",
          "expected_terms": ["산책"],
          "require_disclaimer": false
        }
      ]
    },
    {
      "id": "memory_across_dates_day1_seed",
      "speaker_id": "date_memory_user_001",
      "runner": "orchestrator",
      "seed_medications": [],
      "steps": [
        {
          "id": "store_date_sensitive_memory",
          "text": "오늘 저녁 8시에 혈압약을 먹었다고 기억해줘. 그리고 내일 확인해줘.",
          "expected_mode": "memory_only",
          "expected_intent": "smalltalk",
          "expected_terms": ["혈압약", "8"],
          "require_disclaimer": false
        }
      ]
    },
    {
      "id": "memory_across_dates_day2_recall",
      "speaker_id": "date_memory_user_001",
      "runner": "orchestrator",
      "seed_medications": [],
      "steps": [
        {
          "id": "recall_previous_day_memory",
          "text": "어제 내가 저녁에 먹었다고 말한 약이 뭐였지?",
          "expected_mode": "memory_only",
          "expected_intent": "smalltalk",
          "expected_terms": ["혈압약", "8"],
          "require_disclaimer": false
        },
        {
          "id": "ask_memory_based_followup",
          "text": "그럼 오늘도 비슷한 시간에 챙기면 되겠지?",
          "expected_mode": "memory_only",
          "expected_intent": "smalltalk",
          "expected_terms": ["혈압약"],
          "require_disclaimer": false
        }
      ]
    },
    {
      "id": "ocr_dur_medication_guidance_with_smalltalk",
      "speaker_id": "ocr_dur_user_001",
      "runner": "orchestrator",
      "seed_medications": ["와파린정", "아스피린장용정", "오메프라졸캡슐"],
      "steps": [
        {
          "id": "ocr_meds_loaded_plan_request",
          "text": "처방전 사진에서 읽힌 약들이 있는데, 이 약들 기준으로 복용지도를 해줘.",
          "expected_mode": "tool_first",
          "expected_intent": "medication_query",
          "expected_terms": ["와파린", "아스피린", "오메프라졸"]
        },
        {
          "id": "dur_interaction_check",
          "text": "이 약들 같이 먹어도 괜찮은지 DUR 기준으로 조심할 점 알려줘.",
          "expected_mode": "tool_first",
          "expected_intent": "medication_query",
          "expected_terms": ["와파린", "아스피린"]
        },
        {
          "id": "make_simple_daily_plan",
          "text": "너무 어렵지 않게 아침, 점심, 저녁 기준으로 복용 계획을 세워줘.",
          "expected_mode": "tool_first",
          "expected_intent": "medication_query",
          "expected_terms": ["아침", "저녁"]
        },
        {
          "id": "smalltalk_after_med_guidance",
          "text": "약이 많아서 좀 부담돼. 그냥 짧게 격려해줘.",
          "expected_mode": "memory_only",
          "expected_intent": "smalltalk",
          "expected_terms": [],
          "require_disclaimer": false
        },
        {
          "id": "recall_ocr_meds_later",
          "text": "아까 사진에서 읽힌 약 이름 다시 말해줘.",
          "expected_mode": "memory_only",
          "expected_intent": "medication_query",
          "expected_terms": ["와파린", "아스피린", "오메프라졸"]
        }
      ]
    }
  ]
}
````

````

실행은 이렇게 하면 됩니다.

```bash
./odiss/bin/python scripts/validate_backend_live.py \
  --scenario-file scripts/my_odiss_scenarios.md \
  --strict
````

특정 시나리오만 돌리고 싶으면 이렇게 합니다.

```bash
./odiss/bin/python scripts/validate_backend_live.py \
  --scenario-file scripts/my_odiss_scenarios.md \
  --scenario ocr_dur_medication_guidance_with_smalltalk \
  --strict
```

날짜가 지나도 메모리가 유지되는지 확인하는 3번은 조금 다르게 봐야 합니다.
한 번에 돌리면 “날짜가 지남” 자체보다는 **같은 speaker_id의 이전 기억을 다시 불러오는지**를 보는 테스트가 됩니다.

진짜로 날짜 경과를 보려면 이렇게 나눠서 실행하는 게 좋습니다.

첫날:

```bash
./odiss/bin/python scripts/validate_backend_live.py \
  --scenario-file scripts/my_odiss_scenarios.md \
  --scenario memory_across_dates_day1_seed \
  --strict
```

다음날:

```bash
./odiss/bin/python scripts/validate_backend_live.py \
  --scenario-file scripts/my_odiss_scenarios.md \
  --scenario memory_across_dates_day2_recall \
  --strict
```

그리고 다중화자 테스트에서 중요한 점이 하나 있습니다.
지금 스키마는 `speaker_id` 단위로 유저를 나누기 때문에, **한 시나리오 안에서 화자를 계속 바꾸는 방식보다는 시나리오를 여러 개로 나누는 방식**이 안정적입니다.

즉, 이런 식이 좋습니다.

```txt
multi_user_minsu_32_male
multi_user_jiyoon_67_female
long_term_smalltalk_user_001
date_memory_user_001
ocr_dur_user_001
```

각각 다른 사람처럼 취급됩니다.

다만 현재 필드만으로는 “김민수의 기억이 박지윤에게 새어 나오지 않았는지”를 엄밀히 자동 검증하기 어렵습니다.
왜냐하면 지금 보이는 스키마에는 `expected_terms`는 있지만 `forbidden_terms`가 없기 때문입니다.

예를 들어 박지윤 시나리오에서 답변에 `"김민수"`가 나오면 실패해야 하는데, 지금 구조에서는 이걸 자동으로 강제하기 어렵습니다.
그걸 제대로 잡고 싶으면 나중에 이런 필드를 추가하면 좋습니다.

```json
"forbidden_terms": ["김민수", "32살"]
```

그래도 지금 단계에서는 위 시나리오만으로도 다음은 충분히 확인할 수 있습니다.

```txt
1. 신규 유저가 자기소개를 하면 기억하는가
2. speaker_id가 다른 유저를 따로 기억하는가
3. 장기간 대화처럼 이전 생활 맥락을 다시 불러오는가
4. 같은 speaker_id로 날짜가 지난 뒤에도 기억을 불러올 수 있는가
5. OCR로 들어온 약물 seed를 기반으로 DUR/복용지도 흐름을 타는가
6. 약물 상담 후에도 자연스럽게 스몰토킹으로 전환되는가
```

처음에는 `expected_terms`를 너무 빡세게 잡지 않는 게 좋습니다.
특히 약물 상담 쪽은 모델이 표현을 다르게 할 수 있으니 약 이름 중심으로만 잡는 게 안전합니다.
