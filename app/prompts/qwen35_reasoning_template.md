# Qwen3.5 Reasoning Prompt Template

이 문서는 ODISS 추론 엔진 역할을 수행하는 Qwen3.5 fine-tuning 데이터의 assistant 출력 형식을 정의한다.

## 기본 원칙

- assistant 메시지는 항상 `<think>...</think>` 블록으로 시작한다.
- `<think>` 블록은 긴 비공개 추론이 아니라 학습용으로 통제된 구조화 요약만 담는다.
- tool 호출 전 assistant 메시지의 `<think>`에는 의도, 필요한 tool, 안전 정책을 쓴다.
- tool 결과를 받은 뒤 최종 assistant 메시지의 `<think>`에는 tool 결과 요약과 답변 정책을 쓴다.
- 최종 사용자 답변에는 복약 안전 단정 표현을 피하고, 필요한 경우 `정확한 판단은 의사·약사 상담이 필요합니다.` 문구를 포함한다.

## Tool 호출 전 assistant content

```text
<think>
intent: medication_query | drug_identification | supplement_query | emergency | unknown
needed_tools: 필요한 tool 또는 확인할 정보
safety_policy: 복용 가능/불가능 단정 금지, 전문가 상담, 임의 중단 금지 등
</think>
```

예시:

```text
<think>
intent: medication_query
needed_tools: 병용 금기와 노인주의 확인
safety_policy: 출혈 위험 가능성이 있으므로 임의 병용 허용 답변을 피한다.
</think>
```

이 assistant 메시지에는 OpenAI-compatible `tool_calls`를 함께 둔다.

## Tool 결과 이후 최종 assistant content

```text
<think>
tool_result_summary: tool 결과에서 확인된 핵심 근거
answer_policy: 최종 답변에서 지킬 안전 기준
</think>
사용자에게 보여줄 최종 한국어 답변...
```

예시:

```text
<think>
tool_result_summary: 와파린-아스피린 병용 시 출혈 위험과 고령자 출혈 주의가 확인됨.
answer_policy: 복용 가능 단정 금지, 처방자 또는 약사 확인 권고.
</think>
와파린과 아스피린은 함께 복용하면 출혈 위험이 커질 수 있습니다...
```

## Tool 선택 기준

- 의약품 외형만 있고 이름이 불분명하면 `Tool_Get_Drug_Identification`
- 약물 간 병용 위험이면 `Tool_Check_DUR_Combination_Contraindication`
- 65세 이상이면 `Tool_Check_DUR_Geriatric_Caution`
- DUR 대상 품목 기본 확인이면 `Tool_Get_DUR_Basic_Item_Info`
- 소아/영유아 등 특정 연령이면 `Tool_Check_DUR_Age_Specific_Contraindication`
- 복용량 초과 의심이면 `Tool_Check_DUR_Dosage_Caution`
- 장기 복용 기간 확인이면 `Tool_Check_DUR_Duration_Caution`
- 같은 효능군/성분 중복이면 `Tool_Check_DUR_Duplicate_Therapeutic_Class`
- 서방정 분할/마쇄 여부면 `Tool_Check_DUR_Sustained_Release_Caution`
- 임부/임신 가능성이 있으면 `Tool_Check_DUR_Pregnancy_Contraindication`
- 건강기능식품 상세 성분/주의사항이면 `Tool_Get_Health_Supplement_Detail`
- 건강기능식품 품목명 식별/목록 검색이면 `Tool_Search_Health_Supplement_List`
