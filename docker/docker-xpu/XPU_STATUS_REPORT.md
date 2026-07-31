# LlamaFactory XPU Enablement — Status Update
2026-07-29 · Intel B70 (2× GPU) · [PR #1](https://github.com/kahlun/LlamaFactory/pull/1)

---

## Context
LlamaFactory is an open-source LLM fine-tuning framework (SFT, LoRA/QLoRA, DPO, PPO, PiSSA).
We are enabling it to run on Intel XPU hardware and contributing the environment + test support back upstream.

---

## Test Environment — two torch versions

XPU coverage is validated across a two-version matrix. Each path is exercised on the version it needs:

| Torch | Scope | Coverage |
|---|---|---|
| **2.12.1+xpu** (default Docker image) | Single-GPU / non-FSDP | Unit tests + e2e (train, chat, SGLang), bnb 4-bit QLoRA (bnb 0.47.0) |
| **2.13.0+xpu** | FSDP2 + 2-GPU distributed | multi-device, Ulysses CP, FSDP2 SFT training path |

---

## What We Accomplished

XPU is now a first-class test target, and every XPU-runnable test passes on 2× Intel GPU
(**211/211 runnable, 0 failed** = 201 unit + 10 e2e):

- ✅ Unit tests — **201/201 XPU-runnable pass**
- ✅ End-to-end tests — **10/10 pass** (train, chat, SGLang)
- ✅ 2-GPU distributed (torch 2.13) — multi-device + Ulysses context-parallel pass; FSDP2 SFT training
     path proven (loss converges, checkpoint saved — its test xfails only on a 1-GPU config assumption)
- ✅ bitsandbytes 4-bit (QLoRA) — passes on the default image (2/2, confirmed on B70 2026-07-30)
- ✅ Docker image + compose for Intel XPU shipped

> 211 runnable is 100% of what's runnable; 216 collected total = 211 runnable + 1 pure-SW (cpu-only) +
> 4 blocked. The 4 blocked are all **by-design xfails** (upstream/WONTFIX, not XPU gaps): transformers>4.48
> attention refactor, PiSSA init instability, FSDP2 weight-convert, and the FSDP2-SFT 1-GPU config test.

**Key technical wins:**

- Built a reproducible Docker image (Intel DLE 2026.1 base, torch 2.12.1+xpu) + docker-compose
- Fixed a real source bug: distributed backend returned `gloo` for XPU → switched to `xccl` (`helper.py`)
- Enabled XPU across 22 test files via device-agnostic markers (10 newly XPU-runnable this week)
- Proved FSDP2 SFT training completes on 2-GPU XPU (loss converges, checkpoint saved) on torch 2.13
- PR #1 submitted upstream: Docker files + XPU test enablement

---

## Blockers

**1. PyTorch 2.13 FSDP + GPU IPC — upstream Intel issues**
FSDP regression ([torch-xpu-ops #4427](https://github.com/intel/torch-xpu-ops/issues/4427)) and IPC support ([#1678](https://github.com/intel/torch-xpu-ops/issues/1678)).
→ Tracked upstream; workarounds in place (torch 2.13 + full `/dev/dri` mount + `xccl`).

---

## Next Steps

| Priority | Action |
|---|---|
| 🔴 This week | Get PR #1 reviewed / merged upstream |
| 🟡 Next step | Make torch 2.13 the default image (unblocks FSDP 2-GPU) once #4427 lands |
| 🟡 Next step | Add multibackend XPU docs (NPU-style: setup / getting started) |
| 🟡 Backlog | Optional integrations: Flash-Attention, vLLM, SGLang, DeepSpeed |
| 🟡 Backlog | CI/CD on Intel GPU environment |
---
