# M3Proctor (MA-Nano-GraphRAG)

A multimodal memory-QA method for the **MultimodalTalk** dataset. Built on a
naive-RAG backbone, it adds: round-level chunking, session-summary chunks,
PDF text-layer extraction, VLM captions for images, a question-modality
classifier, and modality-aware re-ranking, followed by a two-stage cascade
answerer that escalates to images / PDF page renders only when the text-only
answer is not confident enough.

## Design rationale

| Observation on the dataset | Design response |
|---|---|
| ~96% of images carry an `img_description` | Reuse the text description in the vector store; call the VLM only for the remaining ~4% |
| 100% of PDFs lack a `pdf_description` | Extract the text layer with PyMuPDF and chunk it (academic PDFs have complete text layers) |
| ~99% of `fm` questions reference image-bearing rounds | Always pass the retrieved candidate images to the VLM and let it pick the file name |
| `mr` questions: ~55% image, ~4% PDF, ~41% text-only | Don't blindly attach images; let the modality classifier decide |
| Cross-session questions (ms / tr) need a global view | Add session-summary chunks to mitigate round-level granularity loss |

## Pipeline

```
build_index
 |- for each round    -> round-chunk           (image descriptions + PDF text snippet)
 |- for each session  -> summary-chunk         (LLM-generated mini-timeline)
 |- for each PDF page -> pdf_page-chunk        (text + 1-sentence LLM digest)
                ↓
         sentence-transformers embed
                ↓
            in-memory matrix

retrieve(question)
 |- ModalityClassifier(LLM, cached) -> {needs_image, needs_pdf, needs_chart}
 |- cosine top-(k * over_fetch) over all chunks
 |- re-rank:
        final = cosine
              + alpha_image * needs_image * has_image
              + alpha_pdf   * needs_pdf   * has_pdf
              + alpha_chart * needs_chart * has_chart
              + summary_boost (per summary chunk)

answer(question, items)
 |- Stage 1: text-only over retrieved chunks
 |    +- high confidence -> return
 |    +- low confidence  -> Stage 2
 |- Stage 2: attach retrieved images and/or rendered PDF pages
 |    based on visual_score / pdf_score / chart_score signals
```

## Directory layout

```
m3exam/m3proctor/
├── README.md
├── config.yaml                # unified config
├── run.py                     # entry point
│
├── core/                      # algorithm: index -> retrieve -> answer
│   ├── indexer.py             # round / session-summary / pdf-page chunks
│   ├── modality_classifier.py # query-side LLM classifier (cached)
│   ├── retriever.py           # cosine + modality-aware re-rank
│   ├── answerer.py            # Stage 1 (text) -> Stage 2 (multimodal) cascade
│   └── pipeline.py            # M3ProctorEvaluator orchestrator
│
├── evaluation/                # scoring + reporting
│   ├── metrics.py             # EM / F1 / BLEU-1 / fm-image / aggregation
│   └── report.py              # per-type table + cascade table renderer
│
├── infra/                     # boundary code: I/O, LLM clients, logging
│   ├── dataset.py             # sessions.json / question.json loaders
│   ├── llm_client.py          # SharedLLM with optional dedicated judge endpoint
│   ├── pdf_render.py          # PyMuPDF page rendering / text extraction
│   └── logging_setup.py       # stdout + file logger with tee
│
└── interfaces/                # abstract bases
    └── base_evaluator.py      # BaseEvaluator contract
```

## Running

```bash
# Full run on a dataset
python -m m3exam.m3proctor.run --dataset Alex_Veterinarian

# Debug on the first 20 questions
python -m m3exam.m3proctor.run --dataset Mina_BotanyStudent --max-questions 20

# Skip the no-cascade ablation
python -m m3exam.m3proctor.run --dataset Mina_BotanyStudent --no-ablation-no-cascade

# Export cascade case study (MR questions only by default)
python -m m3exam.m3proctor.run --dataset Noah_BaristaApprentice --export-cascade-case-study

# Use an explicit datasets root (overrides eval.datasets_root in config.yaml)
python -m m3exam.m3proctor.run --dataset Mina_BotanyStudent \
    --datasets-root /abs/path/to/datasets
```

Make sure `eval.datasets_root` in `config.yaml` points at a directory whose
sub-folders are individual datasets, each containing `sessions.json`,
`question.json`, and an `images/` (plus an optional `pdfs/`) directory.

Dependencies: `openai>=1.0`, `sentence-transformers`, `PyMuPDF`, `numpy`,
`pyyaml`. The local embedding model (default `thenlper/gte-base`) is fetched
from HuggingFace on first use.

## Output

```
m3exam/m3proctor/outputs/<dataset>/
    m3proctor/
        results.json                       # per-question record (full cascade)
        results_ablation_no_cascade.json   # same questions, Stage 1 only (if ablation is on)
        summary.json                       # rubric + cascade summary + experiment_no_cascade
    report.txt                             # per-type metrics table + cascade rate table
    cascade_case_study/                    # optional: exported MR cascade-repair cases
        README.txt
        exported_manifest.json
        case_<NNNN>_run<MMM>_.../
            meta.json
            retrieved_context.txt
            stage1_answer_before_cascade.txt
            stage2_final_model_answer.txt
            attachments/...
```

Metrics reported per question type and aggregated: **EM**, **F1**, **BLEU-1**,
**LLM-judge** (five-point: 0 / 0.25 / 0.5 / 0.75 / 1). `fm` and `fj` only
report EM; aggregated F1 / BLEU-1 / LLM-judge exclude `fm` and `fj`.

## Key hyperparameters (config.yaml)

| Parameter | Default | Description |
|---|---|---|
| `m3proctor.alpha_image` | 0.20 | Re-rank weight when `needs_image=True` and chunk has an image |
| `m3proctor.alpha_pdf` | 0.25 | Re-rank weight when `needs_pdf=True` and chunk has a PDF (PDFs are rarer) |
| `m3proctor.alpha_chart` | 0.15 | Re-rank weight when `needs_chart=True` and chunk has chart-like content |
| `m3proctor.over_fetch` | 6 | Candidate pool size = `top_k * over_fetch` |
| `m3proctor.session_summary` | true | Generate a per-session mini-timeline chunk |
| `m3proctor.vlm_caption_missing` | true | Call the VLM to caption images without `img_description` |
| `m3proctor.pdf_vision_max_pages` | 3 | Fallback: max PDF pages rendered to VLM per question when no hit pages |
| `m3proctor.enable_two_stage` | true | Enable the cascade answerer (Stage 1 -> Stage 2) |
| `m3proctor.run_no_cascade_ablation` | true | Additional Stage 1-only pass for comparison |
| `m3proctor.cascade_case_study_export.enabled` | false | Export bundles where Stage 1 failed and Stage 2 succeeded |
| `pdf.text_max_chars` | 4000 | Max characters of PDF text-layer kept per chunk |
