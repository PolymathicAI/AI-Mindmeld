# Batching and determinism

- **Scope:** why results depend on how prompts are batched, what the package does about it, and the measured size
  of the effect on Kimi K2.6 and Gemma 3.

## The design

- **One planner.** Every backend batches a call's prompts with `latent_interconnect.receivers.kernel.plan_batches`
  under a configured `BatchPlan`. The default plan keeps the prompts' given order, sorts them stably by length, and
  packs micro-batches of up to 32,768 padded tokens. The partition is a pure function of the prompts (their token
  counts, in order) and the plan.
- **The plan is configuration.** It changes the numbers (below), so the configs that score or capture carry it in
  their identity. Each result records the digest of the partition it was computed with
  (`latent_interconnect.receivers.kernel.partition_digest`). What a run only observes (devices, timings, memory) stays in its provenance.
- **Generation is declared too.** A generation backend batches a call under its configured
  `latent_interconnect.receivers.base.GenerationPlan` (sequences in flight, tokens per engine step), part of the
  identity of every generation run (`SuiteRunConfig.generation_plan`, stated by generation runs only, as `plan` by
  scoring runs only). The default is concurrent; `SEQUENTIAL_GENERATION`, one prompt at a time, is the engine `hf`
  serves. Its `prompts_per_call` splits a generation run into in-order calls, each committed as it ends, so a killed
  run resumes by whole calls (`latent_interconnect.evaluation.generating`); a call's tail decodes with fewer
  sequences in flight, so the split is part of the plan too. The runner owns the split: a backend checks only the
  engine fields it serves (`hf` takes any `prompts_per_call`), and keeps the whole plan it accepted.
- **Writes follow the plan.** A score call computes its connectors' writes per micro-batch of its plan
  (`latent_interconnect.receivers.base.check_score_request`), so a live connector's memory scales with the configured
  scoring batch, not the call. The plan bounds the
  objects per micro-batch by `budget_tokens` ÷ the shortest prompt, whatever the objects' residue lengths, so it
  does not by itself cap a residue connector's eval memory. A training `loss` call resolves its own micro-batch's
  writes in one connector call (`training.trainer`).
- **One call unit.** The prompts the planner sees together form one call. Each suite runner defines its calls
  deterministically from the item order, because other prompts in the same call change the micro-batches.
- **One head.** Every logit comes from `latent_interconnect.receivers.kernel.readout_logits`. The LLM's body
  runs in its native precision (bf16 for Kimi), and the head in float32, which removes the 1/16 quantization of bf16
  log-odds at a tiny cost.

## Why it matters: measurements on Kimi K2.6 (one A100, `kimi-stream`, bf16)

| check | result |
|---|---|
| the same batching repeated (3 times in one process, plus a second process) | exactly identical at every layer |
| other budgets (1,024 and 4,096 vs 32,768): hidden states | median cosine 0.99995 at layer 4, falling to 0.9935 at layer 20; median relative L2 1 % rising to 11 %; worst cosine 0.975 |
| top-8 expert choice | the chosen expert set changes for 9 % of tokens at layer 1, rising to 47 % at layer 19 |
| Yes/No log-odds, budget 1,024 vs 32,768 | median absolute difference 0.25, at most 0.75 (the spread across prompts is 1.30); 11.3 % of prompt pairs swap order |

Effect on a readout: on 200 items the per-task AUROC moves by about 0.007–0.008 (median) and 0.02 (95th percentile)
between budgets.

These measurements reproduce results for a fixed batch plan. Changing the plan changes bf16 rounding, and Kimi's
mixture-of-experts routing amplifies those differences. That is why the default plan is fixed, and why every
result records its plan.

## Batch composition

On one A100 with `kimi-stream` in bf16, changing other rows' writes changed expert token counts and
routed-expert outputs even for a fixed prompt and write. Keep the batch plan and write content consistent
when comparing runs, since later layers and routing can amplify these rounding differences.

**Consequences for reading results**
- Two runs whose micro-batches hold different write content differ by this execution sensitivity, even for an
  identical prompt and write.
- The item-bootstrap confidence intervals do not include it. Its size depends on the layer and the curve: the
  observed shifts were about .006 AUROC at the peak layers and about .03 at a deep layer, on near-chance curves.
- Use one fixed batch plan and record it with every result.

## Gemma: the padding class (a different mechanism)

Distinct from Kimi's expert token counts above. With `hf` and transformers 4.57.6, a Gemma micro-batch with no padding
gets no 4-D mask (`masking_utils.py`) and runs SDPA with `is_causal` and native GQA; a batch with any padding keeps
an explicit mask and repeated K/V (`sdpa_attention.py`).

- **A100, measured:** identical prompts score differently by padding class (real name prompts: median 0.15, max 1.09
  nats), and are bit-identical within a class, whatever the batch size, length or batch-mates.
- **H200, measured:** the two classes are bit-identical. Kimi (`kimi-stream`, eager):
  unaffected.
- **Reading results:** item-bootstrap intervals omit this execution variability; compare one prompt across runs only
  on one device class.
- **Forced padding for scoring.** `BatchPlan.pad_every_batch` (default off) makes `hf` right-pad every forward pass
  by one masked column, so every batch takes the explicit-mask class; positions, write sites and read positions are
  the prompts' own, and only a padding backend serves it (`kimi-stream` refuses it). The released Gemma scoring
  suites set it (`configs/release/suites/`). Training, captures, generation and scans keep the default.
- **A fresh scoring run states its padding.** A scoring suite on a backend that pads (`hf`, its class's `pads_every_batch`) is refused by the suite stage when
  its plan leaves `pad_every_batch` out (`run.suite.require_stated_padding`). It states `true` or `false`.
  `kimi-stream` cannot pad (`pads_every_batch = False`, a capability), so a Kimi suite states nothing. The suites of
  `examples/guides/extending.md` §6 state `plan: {pad_every_batch: true}` for this reason.
