# ODISS Engine Call Trace Scenario Suite

이 파일의 JSON 블록만 validate_backend_live.py가 읽습니다.

주의: 현재 하네스는 `id`, `speaker_id`, `runner`, `seed_medications`, `steps[].expected_*`를 주로 검사합니다. `trace_expectations`, `forbidden_terms`는 호출목록/메모리 쓰기 검증을 위한 확장 메타데이터입니다.

```json
{
  "suite_id": "odiss_engine_call_trace_suite_v1",
  "description": "ODISS Conversation/Memory/Reasoning/Tool/Frontier Engine 구조에 맞춘 상세 호출목록 기반 시나리오. validate_backend_live.py는 기본 expected_* 필드를 검사하고, trace_expectations/forbidden_terms는 향후 하네스 확장 또는 수동 리뷰용 메타데이터입니다.",
  "scenarios": [
    {
      "id": "engine_ce_smalltalk_no_med_contamination",
      "speaker_id": "trace_user_smalltalk_med_context_001",
      "runner": "orchestrator",
      "seed_medications": [
        "타이레놀정",
        "이부프로펜정"
      ],
      "trace_expectations": {
        "purpose": "ConversationEngine이 단순 인사/스몰토킹을 약물 상담으로 오염시키지 않는지 확인",
        "expected_engine_sequence": [
          "ConversationEngine.CE_Input",
          "ConversationEngine.CE_Latency",
          "MemoryEngine.ME_Context",
          "ReasoningEngine.RE_Intent",
          "ConversationEngine.CE_Conversation_Core",
          "ConversationEngine.CE_Response",
          "MemoryEngine.ME_Update"
        ],
        "expected_route": "memory_only",
        "expected_tool_calls": [],
        "must_not_call_tools": [
          "T2.병용금기정보조회",
          "T3.노인주의정보조회",
          "T4.DUR품목정보조회",
          "T13.LLM에이전트검색"
        ],
        "expected_memory_reads": [
          "CurrentUserProfile.md",
          "CurrentRequirement.md",
          "PrescriptionLog.md"
        ],
        "expected_memory_writes": [
          "MedicationLog.md",
          "CurrentRequirement.md"
        ],
        "expected_response_object": {
          "response_type": "smalltalk",
          "requires_tts": true,
          "strip_think_tags": true
        }
      },
      "steps": [
        {
          "id": "plain_greeting_should_not_trigger_medication_guidance",
          "text": "안녕하세요. 오늘은 그냥 인사만 하고 싶어요.",
          "expected_mode": "memory_only",
          "expected_intent": "smalltalk",
          "expected_terms": [
            "안녕"
          ],
          "forbidden_terms": [
            "타이레놀",
            "이부프로펜",
            "병용",
            "출혈",
            "DUR"
          ],
          "require_disclaimer": false,
          "trace_expectations": {
            "expected_engine_sequence": [
              "CE_Input",
              "CE_Latency.filler_or_smalltalk",
              "ME_Context.load_profile",
              "RE_Intent.classify_smalltalk",
              "CE_Conversation_Core.package_smalltalk",
              "CE_Response.tts_ready",
              "ME_Update.append_medication_log_as_smalltalk"
            ],
            "expected_tool_calls": [],
            "expected_memory_writes": [
              "MedicationLog.md",
              "CurrentRequirement.md"
            ]
          }
        }
      ]
    },
    {
      "id": "engine_multi_speaker_register_male",
      "speaker_id": "trace_user_leejunho_54_male",
      "runner": "orchestrator",
      "seed_medications": [],
      "trace_expectations": {
        "purpose": "새로운 남성 화자 등록, 프로필 저장, 이후 회상",
        "expected_engine_sequence": [
          "CE_Input",
          "ME_Context.identity_gate",
          "ME_Parse.extract_patient_profile",
          "ME_Update.write_Patient_md",
          "RE_Intent.classify_profile_registration",
          "CE_Response.confirm_registration",
          "ME_Update.write_flash_profile"
        ],
        "expected_tool_calls": [],
        "expected_memory_writes": [
          "Patient.md",
          "CurrentUserProfile.md",
          "patients/{speaker_id}/history.md",
          "MedicationLog.md"
        ],
        "expected_identity_fields": {
          "name": "이준호",
          "age": 54,
          "sex": "남성",
          "conditions": [
            "고혈압"
          ]
        }
      },
      "steps": [
        {
          "id": "male_new_user_registration",
          "text": "처음 왔어요. 제 이름은 이준호이고 54살 남자예요. 고혈압이 있어서 혈압약 상담을 자주 받고 싶어요.",
          "expected_mode": "memory_only",
          "expected_intent": "smalltalk",
          "expected_terms": [
            "이준호",
            "54",
            "고혈압"
          ],
          "forbidden_terms": [
            "타이레놀",
            "이부프로펜",
            "최서연"
          ],
          "require_disclaimer": false,
          "trace_expectations": {
            "expected_engine_sequence": [
              "CE_Input.receive_stt_text",
              "ME_Context.check_new_patient",
              "ME_Parse.extract_name_age_sex_condition",
              "ME_Update.Patient.md",
              "ME_Update.CurrentUserProfile.md",
              "CE_Response.confirm_identity_registration"
            ],
            "expected_tool_calls": []
          }
        },
        {
          "id": "male_profile_recall",
          "text": "내 이름, 나이, 성별, 기저질환을 확인해줘.",
          "expected_mode": "memory_only",
          "expected_intent": "smalltalk",
          "expected_terms": [
            "이준호",
            "54",
            "남자",
            "고혈압"
          ],
          "forbidden_terms": [
            "최서연",
            "29",
            "임신"
          ],
          "require_disclaimer": false,
          "trace_expectations": {
            "expected_engine_sequence": [
              "ME_Context.load_CurrentUserProfile",
              "ME_RAG.search_Patient_md",
              "RE_Intent.classify_profile_recall",
              "CE_Tone.summarize_for_user",
              "CE_Response.tts_ready"
            ],
            "expected_memory_reads": [
              "Patient.md",
              "CurrentUserProfile.md"
            ],
            "expected_tool_calls": []
          }
        }
      ]
    },
    {
      "id": "engine_multi_speaker_register_female_isolation",
      "speaker_id": "trace_user_choiseoyeon_29_female",
      "runner": "orchestrator",
      "seed_medications": [],
      "trace_expectations": {
        "purpose": "다른 화자 등록 및 이전 화자의 프로필/기억 누출 방지",
        "expected_engine_sequence": [
          "CE_Input",
          "ME_Context.identity_gate",
          "ME_Parse.extract_patient_profile",
          "ME_Update.write_Patient_md",
          "RE_Intent.classify_profile_registration",
          "CE_Response.confirm_registration",
          "ME_Update.write_flash_profile"
        ],
        "expected_tool_calls": [],
        "expected_memory_writes": [
          "Patient.md",
          "CurrentUserProfile.md",
          "patients/{speaker_id}/history.md",
          "MedicationLog.md"
        ],
        "expected_identity_fields": {
          "name": "최서연",
          "age": 29,
          "sex": "여성",
          "conditions_or_cautions": [
            "임신 가능성"
          ]
        }
      },
      "steps": [
        {
          "id": "female_new_user_registration",
          "text": "저는 새 사용자예요. 이름은 최서연이고 29살 여자예요. 임신 가능성이 있어서 약 상담을 조심스럽게 받고 싶어요.",
          "expected_mode": "memory_only",
          "expected_intent": "smalltalk",
          "expected_terms": [
            "최서연",
            "29",
            "임신"
          ],
          "forbidden_terms": [
            "이준호",
            "54",
            "고혈압"
          ],
          "require_disclaimer": false,
          "trace_expectations": {
            "expected_engine_sequence": [
              "ME_Context.check_new_patient",
              "ME_Parse.extract_identity",
              "ME_Update.Patient.md",
              "ME_Update.CurrentUserProfile.md",
              "CE_Response.confirm_registration"
            ],
            "expected_tool_calls": []
          }
        },
        {
          "id": "female_profile_recall_no_cross_leak",
          "text": "내 프로필을 다시 말해줘. 혹시 다른 사람 정보랑 섞이면 안 돼.",
          "expected_mode": "memory_only",
          "expected_intent": "smalltalk",
          "expected_terms": [
            "최서연",
            "29",
            "임신"
          ],
          "forbidden_terms": [
            "이준호",
            "54",
            "고혈압"
          ],
          "require_disclaimer": false,
          "trace_expectations": {
            "expected_engine_sequence": [
              "ME_Context.load_profile_by_speaker_id",
              "ME_RAG.search_only_current_speaker_namespace",
              "CE_Response.profile_recall"
            ],
            "expected_memory_reads": [
              "patients/{speaker_id}/Patient.md",
              "CurrentUserProfile.md"
            ],
            "expected_tool_calls": []
          }
        }
      ]
    },
    {
      "id": "engine_long_term_smalltalk_memory_quality",
      "speaker_id": "trace_user_longterm_daily_life_001",
      "runner": "orchestrator",
      "seed_medications": [],
      "trace_expectations": {
        "purpose": "장기간 일반 대화에서 짧은 스몰토킹, 생활 맥락 저장, 구체 회상 확인",
        "expected_engine_sequence": [
          "CE_Input",
          "CE_Latency",
          "ME_Context",
          "ME_Parse",
          "ME_RAG",
          "RE_Intent",
          "CE_Tone",
          "CE_Response",
          "ME_Update"
        ],
        "expected_tool_calls": [],
        "expected_memory_writes": [
          "CurrentRequirement.md",
          "CurrentManual.md",
          "MedicationLog.md",
          "patients/{speaker_id}/history.md"
        ],
        "expected_flash_memory_updates": [
          "최근 요구사항: 짧은 안부, 약 이야기 최소화",
          "생활 맥락: 오전 7시 20분 산책, 보리차"
        ]
      },
      "steps": [
        {
          "id": "store_daily_life_context",
          "text": "오늘 오전 7시에 20분 산책했고, 커피 대신 보리차를 마셨어. 약 얘기보다는 이런 생활 루틴을 기억해줘.",
          "expected_mode": "memory_only",
          "expected_intent": "smalltalk",
          "expected_terms": [
            "산책",
            "20분",
            "보리차"
          ],
          "require_disclaimer": false,
          "trace_expectations": {
            "expected_memory_writes": [
              "CurrentRequirement.md",
              "patients/{speaker_id}/history.md",
              "CurrentManual.md"
            ],
            "expected_tool_calls": []
          }
        },
        {
          "id": "short_smalltalk_without_medical_disclaimer",
          "text": "오늘은 약 얘기 말고 짧게 안부만 물어봐줘.",
          "expected_mode": "memory_only",
          "expected_intent": "smalltalk",
          "expected_terms": [],
          "forbidden_terms": [
            "타이레놀",
            "이부프로펜",
            "DUR",
            "출혈"
          ],
          "require_disclaimer": false,
          "trace_expectations": {
            "expected_engine_sequence": [
              "RE_Intent.classify_smalltalk",
              "CE_Tone.keep_short",
              "CE_Response.smalltalk"
            ],
            "expected_tool_calls": []
          }
        },
        {
          "id": "recall_daily_life_context",
          "text": "내가 오늘 아침에 뭐 했다고 했지? 구체적으로 말해줘.",
          "expected_mode": "memory_only",
          "expected_intent": "smalltalk",
          "expected_terms": [
            "오전 7시",
            "20분",
            "산책",
            "보리차"
          ],
          "require_disclaimer": false,
          "trace_expectations": {
            "expected_memory_reads": [
              "CurrentRequirement.md",
              "patients/{speaker_id}/history.md"
            ],
            "expected_tool_calls": []
          }
        }
      ]
    },
    {
      "id": "engine_date_memory_day1_seed",
      "speaker_id": "trace_user_date_memory_001",
      "runner": "orchestrator",
      "seed_medications": [],
      "trace_expectations": {
        "purpose": "날짜가 포함된 복약 기억을 영구/휘발 메모리에 저장",
        "simulated_date": "2026-05-12",
        "expected_tool_calls": [],
        "expected_memory_writes": [
          "MedicationLog.md",
          "patients/{speaker_id}/history.md",
          "CurrentRequirement.md"
        ],
        "expected_structured_memory": {
          "date": "2026-05-12",
          "time": "21:00",
          "medication": "로사르탄정",
          "action": "taken"
        }
      },
      "steps": [
        {
          "id": "store_specific_date_medication_memory",
          "text": "2026년 5월 12일 화요일 밤 9시에 로사르탄정을 복용했다고 기록해줘.",
          "expected_mode": "memory_only",
          "expected_intent": "medication_query",
          "expected_terms": [
            "2026년 5월 12일",
            "밤 9시",
            "로사르탄"
          ],
          "require_disclaimer": false,
          "trace_expectations": {
            "expected_engine_sequence": [
              "ME_Parse.extract_date_time_medication",
              "ME_Update.MedicationLog.md",
              "ME_Update.structured_memory"
            ],
            "expected_tool_calls": []
          }
        }
      ]
    },
    {
      "id": "engine_date_memory_day2_recall",
      "speaker_id": "trace_user_date_memory_001",
      "runner": "orchestrator",
      "seed_medications": [],
      "trace_expectations": {
        "purpose": "다음 날짜 실행 시 같은 speaker_id의 이전 복약 기억을 회상",
        "simulated_date": "2026-05-13",
        "expected_tool_calls": [],
        "expected_memory_reads": [
          "MedicationLog.md",
          "patients/{speaker_id}/history.md",
          "structured_memory"
        ]
      },
      "steps": [
        {
          "id": "recall_previous_date_medication",
          "text": "어제 밤에 먹었다고 기록한 약이 뭐였지? 시간도 같이 말해줘.",
          "expected_mode": "memory_only",
          "expected_intent": "medication_query",
          "expected_terms": [
            "로사르탄",
            "밤 9시"
          ],
          "require_disclaimer": false,
          "trace_expectations": {
            "expected_engine_sequence": [
              "ME_Context.load_speaker_memory",
              "ME_RAG.search_medication_log",
              "RE_Intent.classify_memory_recall",
              "CE_Response.memory_based_answer"
            ],
            "expected_tool_calls": []
          }
        }
      ]
    },
    {
      "id": "engine_ocr_dur_full_guidance_with_call_list",
      "speaker_id": "trace_user_ocr_dur_001",
      "runner": "orchestrator",
      "seed_medications": [
        "와파린정",
        "아스피린장용정",
        "오메프라졸캡슐",
        "로사르탄정"
      ],
      "trace_expectations": {
        "purpose": "OCR로 들어온 처방 약물 seed를 DUR/HIRA/복용지도/스몰토킹/메모리 회상까지 연결",
        "expected_seed_memory_writes": [
          "OCRHistory.md",
          "Prescription.md",
          "PrescriptionLog.md"
        ],
        "expected_engine_sequence": [
          "CE_Input",
          "ME_Context",
          "ME_RAG",
          "RE_Intent",
          "DUR_Tool_Execution_Engine",
          "RE_Core_Msg",
          "CE_Tone",
          "CE_Response",
          "ME_Update"
        ]
      },
      "steps": [
        {
          "id": "ocr_result_loaded_confirm_and_store",
          "text": "처방전 OCR 결과가 와파린정, 아스피린장용정, 오메프라졸캡슐, 로사르탄정으로 나왔어. 읽힌 약 이름을 확인하고 처방전 기록으로 저장해줘.",
          "expected_mode": "tool_first",
          "expected_intent": "medication_query",
          "expected_terms": [
            "와파린",
            "아스피린",
            "오메프라졸",
            "로사르탄"
          ],
          "trace_expectations": {
            "expected_engine_sequence": [
              "CE_Input.receive_ocr_text_or_stt_text",
              "MemoryEngine.OCR_Logging",
              "ME_Update.OCRHistory.md",
              "ME_Update.Prescription.md",
              "RE_Intent.plan_medication_identification",
              "DUR_Tool.T4.DUR품목정보조회",
              "RE_Core_Msg.extract_fact_data",
              "CE_Tone.easy_korean",
              "CE_Response.tts_ready"
            ],
            "expected_tool_calls": [
              "T4.DUR품목정보조회"
            ],
            "must_not_call_tools": [
              "T2.병용금기정보조회",
              "T3.노인주의정보조회",
              "T5.특정연령대금기정보조회",
              "T6.용량주의정보조회",
              "T7.투여기간주의정보조회",
              "T8.효능군중복정보조회",
              "T9.서방정분할주의정보조회",
              "T10.임부금기정보조회",
              "T13.LLM에이전트검색"
            ],
            "expected_external_apis": [
              "API_MFDS_DUR"
            ],
            "expected_memory_writes": [
              "OCRHistory.md",
              "Prescription.md",
              "PrescriptionLog.md",
              "MedicationLog.md"
            ]
          }
        },
        {
          "id": "dur_safety_all_categories",
          "text": "이 조합을 DUR 기준으로 병용금기, 노인주의, 특정연령대 금기, 용량주의, 투여기간주의, 효능군중복, 서방정 분할주의, 임부금기까지 확인해줘.",
          "expected_mode": "tool_first",
          "expected_intent": "medication_query",
          "expected_terms": [
            "와파린",
            "아스피린",
            "병용",
            "주의"
          ],
          "trace_expectations": {
            "expected_engine_sequence": [
              "ME_RAG.load_prescription_context",
              "RE_Intent.plan_dur_tasks",
              "DUR_Tool.T2.병용금기정보조회",
              "DUR_Tool.T3.노인주의정보조회",
              "DUR_Tool.T4.DUR품목정보조회",
              "DUR_Tool.T5.특정연령대금기정보조회",
              "DUR_Tool.T6.용량주의정보조회",
              "DUR_Tool.T7.투여기간주의정보조회",
              "DUR_Tool.T8.효능군중복정보조회",
              "DUR_Tool.T9.서방정분할주의정보조회",
              "DUR_Tool.T10.임부금기정보조회",
              "ME_Update.DURLinkageHistory.md",
              "RE_Core_Msg.safety_summary",
              "CE_Tone.patient_friendly"
            ],
            "expected_tool_calls": [
              "T2.병용금기정보조회",
              "T3.노인주의정보조회",
              "T4.DUR품목정보조회",
              "T5.특정연령대금기정보조회",
              "T6.용량주의정보조회",
              "T7.투여기간주의정보조회",
              "T8.효능군중복정보조회",
              "T9.서방정분할주의정보조회",
              "T10.임부금기정보조회"
            ],
            "expected_external_apis": [
              "API_MFDS_DUR"
            ],
            "expected_memory_writes": [
              "DURLinkageHistory.md",
              "Prescription.md",
              "PrescriptionLog.md",
              "MedicationLog.md"
            ]
          }
        },
        {
          "id": "simple_daily_medication_plan",
          "text": "너무 어렵지 않게 아침, 점심, 저녁 기준으로 복용지도를 계획해줘. 위험한 조합은 다시 강조해줘.",
          "expected_mode": "tool_first",
          "expected_intent": "medication_query",
          "expected_terms": [
            "아침",
            "점심",
            "저녁",
            "와파린",
            "아스피린"
          ],
          "trace_expectations": {
            "expected_engine_sequence": [
              "ME_RAG.load_DURLinkageHistory",
              "ME_RAG.load_PrescriptionLog",
              "RE_Intent.plan_guidance",
              "RE_Core_Msg.create_medication_plan",
              "CE_Tone.easy_korean",
              "CE_Response.tts_ready",
              "ME_Update.PrescriptionLog.md"
            ],
            "expected_tool_calls": [
              "reuse_previous_DUR_results_or_T4_if_missing"
            ],
            "expected_memory_reads": [
              "DURLinkageHistory.md",
              "PrescriptionLog.md"
            ],
            "expected_memory_writes": [
              "PrescriptionLog.md",
              "MedicationLog.md",
              "CurrentRequirement.md"
            ]
          }
        },
        {
          "id": "health_supplement_interaction",
          "text": "오메가3 건강기능식품도 같이 먹고 있는데, 와파린이나 아스피린이랑 같이 먹어도 괜찮은지 확인해줘.",
          "expected_mode": "tool_first",
          "expected_intent": "medication_query",
          "expected_terms": [
            "오메가3",
            "와파린",
            "아스피린"
          ],
          "trace_expectations": {
            "expected_engine_sequence": [
              "RE_Intent.detect_health_supplement_query",
              "DUR_Tool.T12.건강기능식품목록조회",
              "DUR_Tool.T11.건강기능식품상세정보조회",
              "ME_RAG.load_medication_context",
              "RE_Core_Msg.supplement_medication_caution",
              "CE_Response.tts_ready",
              "ME_Update.HealthSupplementLog.md"
            ],
            "expected_tool_calls": [
              "T12.건강기능식품목록조회",
              "T11.건강기능식품상세정보조회"
            ],
            "expected_external_apis": [
              "API_Health_Supplement"
            ],
            "expected_memory_writes": [
              "HealthSupplementLog.md",
              "MedicationLog.md"
            ]
          }
        },
        {
          "id": "smalltalk_after_guidance_without_losing_context",
          "text": "약이 많아서 조금 불안해. 지금은 긴 설명 말고 짧게 응원해줘.",
          "expected_mode": "memory_only",
          "expected_intent": "smalltalk",
          "expected_terms": [],
          "require_disclaimer": false,
          "trace_expectations": {
            "expected_engine_sequence": [
              "RE_Intent.classify_smalltalk_with_medical_context",
              "CE_Tone.short_empathy",
              "CE_Response.smalltalk",
              "ME_Update.CurrentRequirement.md"
            ],
            "expected_tool_calls": [],
            "must_not_call_tools": [
              "T2.병용금기정보조회",
              "T4.DUR품목정보조회",
              "T13.LLM에이전트검색"
            ],
            "expected_memory_writes": [
              "CurrentRequirement.md",
              "MedicationLog.md"
            ]
          }
        },
        {
          "id": "recall_ocr_med_names_later",
          "text": "아까 OCR에서 읽힌 처방 약 이름을 전부 다시 말해줘. 다른 약 이름을 섞으면 안 돼.",
          "expected_mode": "memory_only",
          "expected_intent": "medication_query",
          "expected_terms": [
            "와파린",
            "아스피린",
            "오메프라졸",
            "로사르탄"
          ],
          "forbidden_terms": [
            "타이레놀",
            "이부프로펜",
            "알마겔"
          ],
          "trace_expectations": {
            "expected_engine_sequence": [
              "ME_RAG.load_OCRHistory",
              "ME_RAG.load_Prescription",
              "RE_Intent.classify_ocr_memory_recall",
              "CE_Response.list_exact_ocr_medications"
            ],
            "expected_tool_calls": [],
            "expected_memory_reads": [
              "OCRHistory.md",
              "Prescription.md",
              "PrescriptionLog.md"
            ]
          }
        }
      ]
    },
    {
      "id": "engine_simulated_ocr_processed_strict_validation",
      "speaker_id": "trace_user_simulated_ocr_20260512",
      "runner": "orchestrator",
      "seed_medications": [
        "와파린정",
        "아스피린장용정",
        "오메프라졸캡슐",
        "로사르탄정"
      ],
      "trace_expectations": {
        "purpose": "실제 이미지 OCR 정확도가 아니라 OCR 완료 후 약물명 목록이 서버로 들어온 이후의 OCR 기록, 처방전 메모리, DUR, 복용지도, 회상 경로를 검증",
        "expected_seed_memory_writes": [
          "OCRHistory.md",
          "Prescription.md",
          "PrescriptionLog.md"
        ]
      },
      "steps": [
        {
          "id": "simulated_ocr_result_store",
          "text": "처방전 OCR 결과가 와파린정, 아스피린장용정, 오메프라졸캡슐, 로사르탄정으로 나왔어. 이 OCR 결과를 처방전 기록으로 저장해줘.",
          "expected_mode": "tool_first",
          "expected_intent": "medication_query",
          "expected_terms": [
            "와파린",
            "아스피린",
            "오메프라졸",
            "로사르탄"
          ],
          "forbidden_terms": [
            "타이레놀",
            "이부프로펜",
            "알마겔"
          ],
          "trace_expectations": {
            "expected_engine_sequence": [
              "CE_Input.receive_ocr_text_or_stt_text",
              "MemoryEngine.OCR_Logging",
              "ME_Update.OCRHistory.md",
              "ME_Update.Prescription.md",
              "RE_Intent.plan_medication_identification",
              "DUR_Tool.T4.DUR품목정보조회",
              "RE_Core_Msg.extract_fact_data",
              "CE_Tone.easy_korean",
              "CE_Response.tts_ready"
            ],
            "expected_tool_calls": [
              "T4.DUR품목정보조회"
            ],
            "expected_memory_writes": [
              "OCRHistory.md",
              "Prescription.md",
              "PrescriptionLog.md",
              "MedicationLog.md"
            ]
          }
        },
        {
          "id": "simulated_ocr_dur_all_categories",
          "text": "이 OCR 처방 약 조합을 DUR 기준으로 병용금기, 노인주의, 특정연령대 금기, 용량주의, 투여기간주의, 효능군중복, 서방정 분할주의, 임부금기까지 확인해줘.",
          "expected_mode": "tool_first",
          "expected_intent": "medication_query",
          "expected_terms": [
            "와파린",
            "아스피린",
            "병용",
            "주의"
          ],
          "forbidden_terms": [
            "타이레놀",
            "이부프로펜",
            "알마겔"
          ],
          "trace_expectations": {
            "expected_engine_sequence": [
              "ME_RAG.load_prescription_context",
              "RE_Intent.plan_dur_tasks",
              "DUR_Tool.T2.병용금기정보조회",
              "DUR_Tool.T3.노인주의정보조회",
              "DUR_Tool.T4.DUR품목정보조회",
              "DUR_Tool.T5.특정연령대금기정보조회",
              "DUR_Tool.T6.용량주의정보조회",
              "DUR_Tool.T7.투여기간주의정보조회",
              "DUR_Tool.T8.효능군중복정보조회",
              "DUR_Tool.T9.서방정분할주의정보조회",
              "DUR_Tool.T10.임부금기정보조회",
              "ME_Update.DURLinkageHistory.md",
              "RE_Core_Msg.safety_summary",
              "CE_Tone.patient_friendly"
            ],
            "expected_tool_calls": [
              "T2.병용금기정보조회",
              "T3.노인주의정보조회",
              "T4.DUR품목정보조회",
              "T5.특정연령대금기정보조회",
              "T6.용량주의정보조회",
              "T7.투여기간주의정보조회",
              "T8.효능군중복정보조회",
              "T9.서방정분할주의정보조회",
              "T10.임부금기정보조회"
            ],
            "expected_external_apis": [
              "API_MFDS_DUR"
            ],
            "expected_memory_reads": [
              "PrescriptionLog.md",
              "Prescription.md"
            ],
            "expected_memory_writes": [
              "DURLinkageHistory.md",
              "MedicationLog.md"
            ]
          }
        },
        {
          "id": "simulated_ocr_recall_exact_names",
          "text": "아까 OCR에서 읽힌 처방 약 이름을 전부 다시 말해줘. 다른 약 이름을 섞으면 안 돼.",
          "expected_mode": "memory_only",
          "expected_intent": "medication_query",
          "expected_terms": [
            "와파린",
            "아스피린",
            "오메프라졸",
            "로사르탄"
          ],
          "forbidden_terms": [
            "타이레놀",
            "이부프로펜",
            "알마겔"
          ],
          "trace_expectations": {
            "expected_engine_sequence": [
              "ME_RAG.load_OCRHistory",
              "ME_RAG.load_Prescription",
              "RE_Intent.classify_ocr_memory_recall",
              "CE_Response.list_exact_ocr_medications"
            ],
            "expected_tool_calls": [],
            "expected_memory_reads": [
              "OCRHistory.md",
              "Prescription.md",
              "PrescriptionLog.md"
            ],
            "must_not_call_tools": [
              "T2.병용금기정보조회",
              "T4.DUR품목정보조회",
              "T13.LLM에이전트검색"
            ]
          }
        },
        {
          "id": "smalltalk_after_simulated_ocr_without_tool_recall",
          "text": "약이 많아서 조금 불안해. 지금은 긴 설명 말고 짧게 응원해줘.",
          "expected_mode": "memory_only",
          "expected_intent": "smalltalk",
          "expected_terms": [],
          "require_disclaimer": false,
          "trace_expectations": {
            "expected_engine_sequence": [
              "RE_Intent.classify_smalltalk_with_medical_context",
              "CE_Tone.short_empathy",
              "CE_Response.smalltalk",
              "ME_Update.CurrentRequirement.md"
            ],
            "expected_tool_calls": [],
            "must_not_call_tools": [
              "T2.병용금기정보조회",
              "T4.DUR품목정보조회",
              "T13.LLM에이전트검색"
            ],
            "expected_memory_writes": [
              "CurrentRequirement.md",
              "MedicationLog.md"
            ]
          }
        }
      ]
    },
    {
      "id": "engine_ocr_request_when_image_missing",
      "speaker_id": "trace_user_ocr_missing_image_001",
      "runner": "orchestrator",
      "seed_medications": [],
      "trace_expectations": {
        "purpose": "처방전 이미지가 없는 상태에서 OCR을 바로 수행하지 않고 업로드/촬영 요청 또는 CE_Prescription_OCR request를 생성",
        "expected_engine_sequence": [
          "CE_Input",
          "RE_Intent.detect_ocr_request",
          "ReasoningEngine.CE_Prescription_OCR",
          "CE_Response.ask_for_image_or_upload"
        ],
        "expected_tool_calls": [],
        "expected_local_agent_request": "prescription_ocr_request"
      },
      "steps": [
        {
          "id": "ask_to_read_prescription_without_image",
          "text": "처방전 사진을 읽어서 약 이름이랑 주의사항을 알려줘.",
          "expected_mode": "ask_user_clarify",
          "expected_intent": "medication_query",
          "expected_terms": [
            "처방전",
            "사진"
          ],
          "require_disclaimer": false,
          "trace_expectations": {
            "expected_engine_sequence": [
              "RE_Intent.detect_missing_image_for_ocr",
              "CE_Prescription_OCR.request_image_from_LocalAgent",
              "CE_Response.ask_user_to_upload_or_capture"
            ],
            "expected_tool_calls": [],
            "must_not_call_tools": [
              "T2.병용금기정보조회",
              "T4.DUR품목정보조회"
            ]
          }
        }
      ]
    },
    {
      "id": "engine_emergency_alert_path",
      "speaker_id": "trace_user_emergency_001",
      "runner": "orchestrator",
      "seed_medications": [
        "아스피린장용정"
      ],
      "trace_expectations": {
        "purpose": "응급 발화가 들어오면 약물 상세설명보다 응급 안내/알림 경로가 우선되는지 확인",
        "expected_engine_sequence": [
          "CE_Input",
          "RE_Intent.detect_emergency",
          "ToolAPI.emergency_alert_or_safety_response",
          "CE_Response.emergency_tts",
          "ME_Update.MedicationLog.md"
        ],
        "expected_priority": "emergency_over_dur",
        "expected_tool_calls": [
          "emergency_alert"
        ]
      },
      "steps": [
        {
          "id": "emergency_after_medication",
          "text": "아스피린 먹고 나서 숨이 차고 얼굴이 붓고 쓰러질 것 같아.",
          "expected_mode": "frontier_first",
          "expected_intent": "emergency",
          "expected_terms": [
            "119",
            "응급",
            "즉시"
          ],
          "trace_expectations": {
            "expected_engine_sequence": [
              "RE_Intent.detect_emergency_signal",
              "RE_Core_Msg.emergency_instruction",
              "CE_Tone.clear_urgent_voice",
              "CE_Response.requires_tts_true",
              "ME_Update.MedicationLog.md"
            ],
            "expected_tool_calls": [
              "emergency_alert"
            ],
            "must_not_wait_for": [
              "DUR full lookup completion",
              "LLM long reasoning"
            ]
          }
        }
      ]
    }
  ]
}
```
