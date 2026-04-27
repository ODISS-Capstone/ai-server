# Qwen3.5 LongCOT Training Template

이 문서는 Qwen3.5 reasoning-engine fine-tuning 데이터에서 사용하는 `<think>...</think>` longCOT 템플릿이다.

런타임 프롬프트와 혼동하지 않는다.

- 런타임 프롬프트: `app/prompts/qwen35_reasoning_template.md`
- 훈련용 longCOT 템플릿: 이 파일과 `data/fine_tuning/qwen35_jsonl_guide.md`

## Tool 호출 전 LongCOT

```text
<think>
1. 의도: medication_query | drug_identification | supplement_query | emergency | unknown
2. 입력 근거: 사용자 질문, OCR 결과, 복약 맥락에서 판단에 필요한 정보를 요약한다.
3. 위험 후보: 병용 금기, 고령자 주의, 특정 연령 금기, 용량/기간 주의, 중복 처방, 서방정 분할, 임부 금기, 건강기능식품 상호작용 가능성을 검토한다.
4. 필요한 API: 사용할 tool 이름과 선택 이유를 쓴다.
5. tool 인자 결정: item_name, item_seq, product_name, print_front 등 schema에 맞는 인자를 입력 근거에서 고른다.
6. 답변 보류: API 결과 확인 전에는 복용 가능 여부를 단정하지 않는다.
</think>
```

이후 같은 assistant 메시지에 `tool_calls`를 붙인다.

## Tool 결과 후 LongCOT

```text
<think>
1. tool 결과 요약: 반환된 items, 주의 문구, 금기 여부, 성분/품목 정보를 요약한다.
2. 근거 평가: 결과가 질문의 위험 판단에 어떤 의미인지 판단한다.
3. 답변 전략: 사용자에게 쉬운 표현으로 무엇을 해야 하는지 안내한다.
4. 안전 문구: 임의 복용, 임의 중단, 처방 변경 지시를 피하고 전문가 상담 문구를 포함한다.
</think>
```

이후 같은 assistant 메시지에 사용자에게 보여줄 최종 답변을 이어 쓴다.

## 예시

```text
<think>
1. 의도: medication_query
2. 입력 근거: 사용자는 와파린 복용 중이고 아스피린 장용정을 새로 처방받아 병용 가능 여부를 묻고 있다.
3. 위험 후보: 항응고제와 아스피린 병용은 출혈 위험이 증가할 수 있으며, 72세 고령자라 이상반응 위험 확인이 필요하다.
4. 필요한 API: 병용 금기 확인을 위해 Tool_Check_DUR_Combination_Contraindication, 고령자 주의 확인을 위해 Tool_Check_DUR_Geriatric_Caution을 호출한다.
5. tool 인자 결정: 새로 처방받은 약인 아스피린 장용정을 item_name으로 사용한다.
6. 답변 보류: DUR 결과 확인 전에는 같이 먹어도 된다고 답하지 않는다.
</think>
```
