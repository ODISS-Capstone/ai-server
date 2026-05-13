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

- Identity/profile registration: closed for orchestrator-level validation.
  `EngineOrchestrator.run_turn(..., run_identity_gate=True)` now runs the
  identity gate before normal RE/tool flow and returns the gate result in the
  pipeline contract. `validate_backend_live.py` uses this path for
  `expect_identity_gate` steps.
- Date-specific medication memory: closed for explicit taken-dose records.
  `MemoryEngine.update_and_compress()` now extracts typed medication events
  such as `2026-05-12 21:00 로사르탄정`, stores them under the speaker patient
  namespace, syncs structured memory, and allows next-turn recall through
  `medication_events.md`.
- Daily-life smalltalk memory: generic history and flash requirement updates
  exist, but `CurrentManual.md` and structured daily-routine extraction are not
  yet first-class outputs.
- OCR text ingestion: closed for explicit STT OCR result text. `MemoryEngine`
  now normalizes utterances such as "OCR 결과가 A, B로 나왔어" into
  `OCRHistory.md`, `Prescription.md`, `PrescriptionLog.md`, and speaker-scoped
  structured medication memory. The WebSocket OCR image path remains separate
  and still performs pending-confirmation before saving.

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
