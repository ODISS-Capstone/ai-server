# ODISS Engine Call Trace Gap Triage

This note tracks scenario expectations that are now visible through the
trace-gate harness but require behavioral upgrades before every authored
scenario can pass as a hard gate.

## Policy Alignment

- Emergency route: current runtime treats emergency as deterministic
  `tool_first` policy (`emergency_alert`) because safety decisions must be
  auditable. The scenario `engine_emergency_alert_path` currently expects
  `frontier_first` at the step level. Recommended alignment: keep runtime
  `tool_first` and update the scenario expectation when scenario files are
  intentionally revised.
- Missing prescription image: `engine_ocr_request_when_image_missing` expects
  image-request behavior. Current route can mark `ask_user_clarify`, but the
  explicit LocalAgent OCR request path should be promoted into the
  orchestrator result contract before making this a hard scenario gate.

## Memory Behavior Gaps

- Identity/profile registration: `identity_guard.py` supports registration
  and confirmation, but `EngineOrchestrator.run_turn()` does not currently run
  the identity gate before normal CE/ME/RE flow. Live validation can call the
  identity gate separately with `expect_identity_gate`, but full scenario
  parity requires an orchestrator option or wrapper.
- Date-specific medication memory: scenarios expect extraction of date/time
  medication events such as `2026-05-12 21:00 로사르탄정`. Current memory
  update stores the turn text and short history, but does not yet create a
  typed medication event record for robust next-day recall.
- Daily-life smalltalk memory: generic history and flash requirement updates
  exist, but `CurrentManual.md` and structured daily-routine extraction are not
  yet first-class outputs.
- OCR text ingestion: HTTP/WebSocket OCR paths write OCR history and
  prescription logs, but text-only STT instructions such as "OCR 결과가 A, B로
  나왔어" are not yet normalized into `OCRHistory.md`, `Prescription.md`, and
  `PrescriptionLog.md` by the orchestrator.

## Tool Coverage Gaps

- DUR full categories: `dur_check` maps to T2-T10 trace IDs once executed, but
  scenario assertions should remain report-only until live API credentials and
  deterministic mock fixtures exist.
- Health supplement flow: `supplement_lookup` maps to T11/T12 trace IDs, but
  medication-supplement interaction quality depends on tool response fixtures
  or live external API availability.
- HIRA/pill identification: `hira_lookup` is traced as `HIRA.의약품식별조회`;
  scenario files should use this id or the validator alias table should be
  extended if a T-code is assigned later.

## Gate Promotion Recommendation

1. Keep unit-level trace schema and validator tests as CI hard gates.
2. Keep `validate_backend_live.py --scenario-file ...` as report-only until
   the above behavior gaps are closed.
3. Promote one scenario family at a time to hard gate:
   smalltalk contamination -> memory recall -> OCR/missing-image ->
   DUR/tool-first -> emergency.
