# NeoCortex-X Stage 6 Evidence Packets Plan

Дата: 2026-06-24.
Design lineage:
- `🧠 NEOCORTEX-X.md`
- `docs/superpowers/specs/2026-06-23-neocortex-x-core-experiment-design.md`
- `docs/superpowers/plans/2026-06-24-neocortex-x-completion-audit-and-next-gaps.md`

## Цель

Stage 6 превращает проверенные Stage 5/5B external observations в компактные
evidence packets для долговременной памяти NeoCortex-X.

Это первый практичный шаг после завершенного core experiment:

```text
external observations -> provenance state -> evidence packet -> memory/world model
```

Stage 6 не делает live web search и не заявляет AGI/Phi. Он задает устойчивый
формат, который позже сможет принимать live-search backend, CodePy cache,
world-model planner и compact AI-native memory keys.

## Packet Contract

Каждый packet обязан содержать:

- stable `packet_id`;
- compact `memory_key`;
- `subject`;
- выбранное factual `value`;
- confidence из trust-weighted provenance;
- supporting provenance claims;
- rejected conflicting values;
- contradiction flag;
- replay hash.

## Controls

- Packet создается только из provenance adapter state, а не из latest-only
  baseline.
- Packet must replay deterministically from the cached fixture.
- Packet must preserve rejected values, so memory does not erase contradictions.
- Packet keys must be unique and compact enough for later CodePy/world-model use.

## Pass/Fail

- `protocol_evidence.valid = true`;
- fixture valid and non-empty;
- packet count equals fixture item count;
- factual accuracy >= `0.95`;
- provenance coverage = `1.0`;
- deterministic replay = `true`;
- compact key uniqueness = `true`.

## Files

- `research/neocortex_x/stage6_evidence_packets.py`
- `Makefile`
  - add `neocortex-x-stage6-packets-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage6_evidence_packets.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage6_evidence_packets.py
make neocortex-x-stage6-packets-smoke
```
