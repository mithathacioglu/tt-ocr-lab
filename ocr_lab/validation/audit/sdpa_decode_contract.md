# `sdpa_decode` shape contract — audit for claim B3

NewMind reports that swapping the prefill `sdpa` op for `scaled_dot_product_attention_decode` (flash decode) on the same KV cache + Q tensors produced gibberish output (`"换 head\n换 head\n..."`) without any assertion firing.

## What the device op actually expects

[/home/aroberge/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp](/home/aroberge/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp), unpaged-causal branch:

```243:267:tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp
        // Assert we are in decode mode if not causal
        // Q: [1, B, NH, D]
        // K: [1, B, S, D]
        // V: [1, B, S, D]

        // Batch must match
        const auto B = q_shape[1];
        if (operation_attributes.share_cache.value_or(false)) {
            TT_FATAL(k_shape[0] == 1, "Share cache expects K to have batch size of 1, but got {}", k_shape[0]);
            TT_FATAL(v_shape[0] == 1, "Share cache expects V to have batch size of 1, but got {}", v_shape[0]);
        } else {
            TT_FATAL(k_shape[0] == B, "K tensor batch size ({}) must equal B ({})", k_shape[0], B);
            TT_FATAL(v_shape[0] == B, "V tensor batch size ({}) must equal B ({})", v_shape[0], B);
        }
        ...
        TT_FATAL(q_shape[0] == 1, "Q tensor batch size must be 1 for decode mode but got {}", q_shape[0]);

        TT_FATAL(
            k_shape[-2] == v_shape[-2],
            "K and V tensors must have the same sequence length. K: {}, V: {}",
            k_shape[-2],
            v_shape[-2]);
```

So the contract is:

| dim | Q | K | V | mask |
|-----|---|---|---|------|
| 0 | 1 | B (or 1 if `share_cache`) | B (or 1 if `share_cache`) | 1 or B |
| 1 | B (= n_decode_users) | n_kv_heads | n_kv_heads | 1 or B |
| 2 | NH (n_q_heads) | S (max cache seq) | S | NH (heads) |
| 3 | D (head_dim) | D | D | S |

That is: the n_q_heads of Q live in **dim 2**, n_kv_heads of K/V live in **dim 1**, and the (single) decode position lives implicitly in `cur_pos_tensor`/`cur_pos`. Q is *not* batched in dim 0 like prefill SDPA would.

## What NewMind's prefill-style layout looks like

The existing [ocr_lab/ttnn_decoder_hybrid.py](tt-ocr-lab/ocr_lab/ttnn_decoder_hybrid.py) builds, for a single decode step:

- Q after `nlp_create_qkv_heads_decode`: `(1, B=1, n_heads=12, head_dim=128)` — i.e. shape `[1, 1, 12, 128]`.
- KV cache populated via `ttnn.fill_cache`: `(1, n_kv_heads=2, max_cache, head_dim=128)`.

Match against the contract above:

- `q_shape = [1, 1, 12, 128]` → `B = q_shape[1] = 1`. OK by line 260 (`q_shape[0] == 1`).
- `k_shape = [1, 2, max_cache, 128]` → `k_shape[0] = 1`. The `else` branch at line 254 then checks `k_shape[0] == B` → `1 == 1` ✓.
- `v_shape = [1, 2, max_cache, 128]` → `v_shape[0] = 1`. ✓
- GQA check at line 308: `k_shape[1] = 2 > 1` → `is_gqa = true`. Then line 316: `q_shape_unpadded[2] % k_shape[1] == 0` → `12 % 2 == 0` ✓.

**All validators pass with NewMind's input.** This is why no `TT_FATAL` fires.

## So why the gibberish?

The shape contract is satisfied but the **semantic** contract is not:

- The op assumes `B = q_shape[1]` is the **batch of decode users** (different sequences whose cur_pos is independent), and that `n_kv_heads = k_shape[1]` are the GQA heads to be replicated across the `q_shape[2] = NH` query heads.
- With `B = 1`, only **one** position in K is read per call, parameterized by `cur_pos[0]` (or `cur_pos_tensor[0]`).
- If `cur_pos` was inherited from the prefill loop and points to (or beyond) an uninitialized cache slot — or if it indexes into the cache at the wrong offset — the op will silently read garbage.

A second silent failure mode: the unpaged decode requires `k_shape[2] % k_chunk_size == 0` (line 276). If the caller's `max_cache` is tile-aligned to 32 but `k_chunk_size` defaults from `get_chunk_size(s)` to e.g. 256, this check fires loudly. But if `k_chunk_size = 0` (paged path), or the chunk happens to evenly divide `max_cache`, the chunk validator passes and the op runs over the entire cache including uninitialized tail tiles.

A third silent failure mode: the wrapper `ExecuteScaledDotProductAttentionDecode::invoke` ([sdpa_decode.cpp:35-92](/home/aroberge/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.cpp)) **does not validate the relationship between `cur_pos` and `K` seq length** beyond the device-side `cur_pos_val < k_shape[-2]` check at line 298 — which is satisfied as long as `cur_pos < max_cache`, regardless of whether the cache was actually filled up to `cur_pos` or not.

## Missing validators (candidates to upstream)

1. **No check that `cur_pos > 0` matches non-zero K content.** The op trusts that the caller has filled the cache up to position `cur_pos[i]`. A silent mismatch (filling 0..N-1 but passing `cur_pos = N+M`) reads zeros and produces zero-attention-weighted gibberish.
2. **No check that Q's `n_q_heads` dim placement is consistent with caller intent** when both `q_shape[1] = 1` and `q_shape[2] = n_heads` *or* `q_shape[1] = n_heads, q_shape[2] = 1` would technically pass the rank-4 layout check (the second triggers the `q_shape[0] == 1` check pass and then GQA check on `q_shape_unpadded[2]` evaluates `1 % nkv == 0` only for `nkv = 1`, otherwise it fires — so this is partially caught).
3. **No guard on `cur_pos_tensor.shape[-1] == B`** in the unpaged branch (only enforced for the paged branch at line 173).

## Verdict

**CONFIRMED that no assertion fires** for the layout NewMind used (Q `[1,1,12,128]`, K/V `[1,2,max_cache,128]`, cur_pos as scalar). The shape contract is technically satisfied; the semantic mismatch is most likely either (a) `cur_pos` interpretation (decode reads from the wrong slot of the cache), or (b) head-fusion mismatch between how the prefill `nlp_create_qkv_heads` lays out Q vs. what the flash-decode kernel expects (the decode kernel may fuse repeated GQA heads across cores assuming a different stride).

The Phase 2 probe `probe_sdpa_decode_contract.py` will pin down which of (a)/(b)/(c) is the actual root cause and what the minimum repro tensor shapes are.

This audit is **the most actionable upstream item**: every "silent failure mode" listed above is a candidate `TT_FATAL` or `TT_LOG_WARNING` to file as a tt-metal issue.
