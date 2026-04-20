---
name: dots.mocr 5/5 keyword pipeline
description: End-to-end OCR pipeline for dots.mocr producing 5/5 Turkish keyword match deterministically at native resolution
type: project
originSessionId: 1a0bff9f-b343-4573-a4ea-fc4f43939b6f
---
## Best known 5/5 config (2026-04-15)

`GREEDY=1 NATIVE=1 HF_DECODE=1 MAX_SEQ=8192 N_TT=11 python ocr_lab/run_hybrid_fast.py`

- **108.5s per page**, 5/5 keywords deterministic
- Output: "hükümler, Davacı 'e ve davalı 'e 18/01/2018 tarihinde tebliğ olunmuş... Dilekçesi ile hükmün... kesinleştiği tasdik olunur"

## Latency breakdown
- TT vision (11 blocks): ~0.9s
- CPU vision (31 blocks at seq=2791): ~76s ← main bottleneck
- Decoder prefill (HF fp32): ~8s
- HF decode (178 tokens greedy): ~23s (128ms/tok)

## Why configs must be exactly these
- **NATIVE=1**: fixed_page resize drops resolution → "hüküml" instead of "hükümler". Native image produces seq=2791 and correct OCR.
- **HF_DECODE=1**: TT decode works for ~30 steps, then cumulative bfloat16 drift causes divergence (confirmed via `ocr_lab/probe_tt_decode_divergence.py` — first HF/TT argmax divergence at step 39, 10/179 total). HF decode at seq=2791 preserves 5/5.
- **N_TT=11**: 10→5/5 (112.9s), 11→5/5 (108.5s), 12→4/5, 15→4/5. bfloat8_b accumulation in TT vision blocks 12+ degrades "hükümler" word detection. 11 is sweet spot.
- **GREEDY=1**: sampling (temp=0.1, top_p=0.9, freq_penalty=0.03) non-deterministic, sometimes 4/5 sometimes 5/5. Greedy argmax stable.

## Known dead-ends attempted
- Lower resolution (672, 1008): resolution-limited to 4/5
- Higher resolution via fixed_page (1120+, MAX_SEQ=4096): TT vision garbage (Chinese chars, hallucinations)
- torch.compile vision blocks: JIT overhead beats savings for single-run
- bf16 HF model: this CPU lacks AVX512-BF16/AMX, bf16 slower than fp32
- Mask shape (1,1,1,N) "fix": SDPA asserts because Q is (1,12,1,128) but reshape to tile makes Q logical seq=32; original (1,1,32,N) mask is correct

## Path to 5-8s target (not achieved)
Requires:
1. **TP vision** across mesh (2x4 or 1x4): 31 CPU blocks → TT blocks with sharded weights. Previous attempt blocked on ShardTensorToMesh + nlp_create_qkv_heads shape semantics (unexpected rank-5 cos, 2× seq). See deepseek_v3/mla1d.py as reference pattern.
2. **TT decode precision fix**: cumulative bf16 drift at seq>2000. Would need higher precision KV cache or different SDPA config. Current HiFi4/fp32_dest_acc already max.
