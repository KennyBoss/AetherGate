# AetherGate — the Law of Routing Existence & the Headroom Meter

This branch (`hybrid-router`) carries a self-contained research-to-product line.
Everything lives under **`research/hybrid_router/`**.

## TL;DR

> **Routing/ensembling/hybridizing two systems creates value if and only if they
> fail on orthogonal inputs.** Identifiability sets how much of that value a router
> captures; model capacity is irrelevant.
>
> `routing_advantage = headroom(error_orthogonality) × efficiency(identifiability)`

Proven on stand-in and **real trained models** (context-capped Transformer vs
KV-memory SSM), through deterministic, multi-seed sweeps with three built-in
anti-illusion guards.

## Start here

| | path |
|---|---|
| **The law** | `research/hybrid_router/LAW_OF_ROUTING_EXISTENCE.md` |
| **The product** (CLI tool) | `research/hybrid_router/headroom_meter/` |
| Full line + results | `research/hybrid_router/README.md` |
| Law→scale plan + real-model stages | `research/hybrid_router/ROADMAP_LAW_TO_SCALE.md` |

## The Headroom Meter (run it today)

Decide whether to combine two systems in seconds — no training, no weights:

```bash
cd research/hybrid_router/headroom_meter
echo '{"a":[1,1,1,0,1],"b":[0,0,1,1,0]}' | python3 headroom_meter.py --stdin
# -> COMBINE / DEPLOY-DOMINANT / DON'T-COMBINE  + headroom, orthogonality
```

## Honest framing

Not "SSM beats Transformers." The measured claim: *for long-delay retrieval a
memory-augmented SSM holds constant state cost, while a Transformer's cost grows
with the required context length* (Stage 3 efficiency frontier). The law is the
filter that told us when **not** to build a router — and saved the months.

Release tags: `hybrid-router-v2.0` (law), `hybrid-router-v2.1-headroom-meter` (tool).
