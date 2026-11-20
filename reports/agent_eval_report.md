# Agent evaluation report (offline-planner, as of 2026-11-12)

| Metric | Value |
| --- | --- |
| n_questions | 20 |
| tool_selection_accuracy | 1.0 |
| tool_recall | 1.0 |
| answer_rate | 1 |
| source_recall | 0.971 |
| source_precision | 1.0 |
| key_fact_recall | 1.0 |
| abstention_accuracy | 1 |
| structured_output_rate | 1.0 |
| dropped_source_rate | 0 |
| mean_steps | 1.25 |
| mean_latency_ms | 1.2 |

## Questions to review

| id | question | expected tools | used tools | source recall | source precision |
| --- | --- | --- | --- | --- | --- |
| a13 | For DEV-2026-0071, what does the change control procedure say about implementing a change before it is approved? | register, sops | register, sops | 0.50 | 1.00 |
