<h1 align="center"> 💫 M<sup>3</sup>Exam: Benchmarking Multimodal Memory for Realistic User-Agent Interactions </a></h2>

<div align="center">
    <a href="https://arxiv.org/abs/2601.03515">
    <img src="https://img.shields.io/badge/📃%20arXiv-Paper-b31b1b.svg"></a>
    <a href="https://huggingface.co/datasets/Ethan-Bei/Mem-Gallery">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg">
</div>

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<h5 align="center">

<img src="figures/main.png">

</h5>

This is the official project repository for **M<sup>3</sup>Exam**.

M<sup>3</sup>Exam is a novel query-centric multimodal conversational QA benchmark built on realistic user-agent interactions, enabling balanced multi-dimensional evaluation across multimodal memorizing, cross-modal reasoning, and implicit-intent interpreting over long-horizon histories of dialogue, images, and documents. We further propose **M<sup>3</sup>Proctor**, a modality-aware multimodal memory method that detects query modality bias and escalates to raw visual sources only on demand through a cost-aware cascade, enabling efficient multimodal evidence management with selective rather than indiscriminate visual injection.

## 📚 Overview

Each item in M<sup>3</sup>Exam is a **persona** — a long-horizon, multi-session history between a user and an assistant, grounded by a profession/hobby-driven *core event* timeline and interleaved with the images and PDF documents the user shares along the way. On top of this history we annotate questions that probe **eight** complementary memory and reasoning abilities:

| Code | Type | What it tests |
| ---- | ---- | ------------- |
| `SS` | Single Session | Recall a fact stated within one session. |
| `MS` | Multi Session | Synthesize consistent evidence across several sessions. |
| `TR` | Temporal Reasoning | Reason over dates and the ordering of events. |
| `TH` | Thematic / Case Management | Bundle scattered advice into a complete, coherent answer. |
| `II` | Implicit Inference | Infer implicit user intent, posture, or state from the dialogue. |
| `MR` | Multimodal Reasoning | Reason over shared images, charts, and PDF documents. |
| `FM` | Find Matching Image | Retrieve the exact image file the user once shared. |
| `FJ` | Factual Judgement (MCQ) | Pick the correct option in a multiple-choice factual check. |

Every question is paired with an answer (free-form answers carry multiple acceptable surface forms) and `supporting_facts` — the dialogue round IDs (e.g. `D2:1,D12:1`) that ground it.

## 🔧 Requirements

M<sup>3</sup>Exam targets **Python 3.10+**. Install the dependencies:

```bash
pip install openai pyyaml sentence-transformers PyMuPDF numpy
```

- The data-generation and evaluation pipelines call an OpenAI-compatible chat API (text + vision). Configure your endpoints in [config/config.yaml](config/config.yaml).
- M<sup>3</sup>Proctor additionally uses a local HuggingFace embedding model (default `thenlper/gte-base`), fetched automatically on first run.

## 📦 Dataset

A ready-to-inspect example persona lives under [example_set/](example_set/). Each persona directory has the following layout:

```
<Persona_Name>/
├── sessions.json                 # multi-session dialogue history (the memory)
├── question.json                 # annotated questions over the history
├── timeline_<Persona_Name>.json  # the core-event timeline anchoring generation
├── images/                       # images shared in the dialogue (img_<n>.jpg/png)
└── pdfs/                         # PDF documents shared in the dialogue
```

### Data Format

**`sessions.json`** — a list of dated sessions, each holding a list of dialogue rounds:

```json
[
  {
    "session_id": "D1",
    "date": "2023-04-01",
    "dialogues": [
      {
        "round": "D1:1",
        "user": "I just started working at an amazing specialty coffee shop ...",
        "assistant": "Congratulations on your new apprenticeship! For a classic latte ...",
        "img_file": ["img_1.jpg", "img_2.jpg"],
        "img_id": [1, 2],
        "img_description": "A photo of the coffee bar menu at a specialty coffee shop ..."
      }
    ]
  }
]
```

Rounds that reference a document carry the analogous `pdf_file` field; rounds without attachments simply omit the visual keys.

**`question.json`** — a flat list of annotated questions:

```json
[
  {
    "question": "Find the very first photo Noah shared — a café chalkboard menu ...",
    "answer": ["img_1.jpg"],
    "supporting_facts": "D1:1",
    "type": "fm",
    "label": "img_1.jpg"
  }
]
```

## 🚀 Get Started

The full data pipeline runs as four stages, each exposed as a module under [execution/](execution/). All stages read their parameters from [config/config.yaml](config/config.yaml); the most common knobs can also be overridden on the command line.

**1. Generate the core-event timeline** that anchors a persona:

```bash
python -m m3exam.execution.run_timeline
```

**2. Generate questions** of a given type over the dialogue history:

```bash
# Generate 50 multimodal-reasoning questions (overrides config.yaml)
python -m m3exam.execution.run_questions --type MR --num 50 \
    --dialogue-route example_set/Noah_BaristaApprentice \
    --output-dir example_set/Noah_BaristaApprentice
```

`--type` accepts any of `SS, MS, TR, TH, II, MR, FM, FJ`.

**3. Finalize** a generated persona into the released dataset layout:

```bash
python -m m3exam.execution.run_finalize
```

**4. Evaluate** a model (or memory baseline) against the questions:

```bash
python -m m3exam.execution.run_evaluation
```

Set `evaluation.evaluation_type` (`text` or `multimodal`), the data/question/result directories, and the evaluation/judge model endpoints under the `evaluation:` block of [config/config.yaml](config/config.yaml).

### Evaluation Metrics

Metrics are reported per question type and aggregated: **EM**, **F1**, **BLEU-1**, and a five-point **LLM-judge** (0 / 0.25 / 0.5 / 0.75 / 1). `FM` and `FJ` report exact match only; aggregated F1 / BLEU-1 / LLM-judge exclude `FM` and `FJ`.

## 🧠 M<sup>3</sup>Proctor

M<sup>3</sup>Proctor is our modality-aware memory method, packaged under [m3proctor/](m3proctor/). Built on a naive-RAG backbone, it adds round-level chunking, session-summary chunks, PDF text-layer extraction, VLM captions for images, a query-side modality classifier, and modality-aware re-ranking, followed by a two-stage cascade answerer that escalates to images / rendered PDF pages **only when the text-only answer is not confident enough**.

```bash
# Full run on one persona
python -m m3exam.m3proctor.run --dataset Noah_BaristaApprentice

# Debug on the first 20 questions
python -m m3exam.m3proctor.run --dataset Noah_BaristaApprentice --max-questions 20
```

See [m3proctor/README.md](m3proctor/README.md) for the full pipeline description, directory layout, and hyperparameters.

## 🧰 Experimental Settings

We have incorporated several baseline methods and benchmark datasets:

| Baseline | Paper | Code |
| -------- | ----- | ---- |
| NaiveRAG | [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) | [nano-graphrag](https://github.com/gusye1234/nano-graphrag) |
| A-Mem | [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110) | [A-Mem](https://github.com/WujiangXu/A-mem) |
| Mem0  | [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413) | [Mem0](https://github.com/mem0ai/mem0) |
| MemoryOS | [Memory OS of AI Agent](https://aclanthology.org/2025.emnlp-main.1318.pdf) | [MemoryOS](https://github.com/BAI-LAB/MemoryOS) |
| UniversalRAG | [UniversalRAG: Retrieval-Augmented Generation over Corpora of Diverse Modalities and Granularities](https://arxiv.org/abs/2504.20734) | [UniversalRAG](https://github.com/wgcyeo/UniversalRAG) |
| RAG-Anything | [RAG-Anything: All-in-One RAG Framework](https://arxiv.org/abs/2510.12323) | [RAG-Anything](https://github.com/HKUDS/RAG-Anything) |
| MIRIX | [MIRIX: Multi-Agent Memory System for LLM-Based Agents](https://arxiv.org/abs/2507.07957) | [MIRIX](https://github.com/Mirix-AI/MIRIX) |
| MemVerse | [MemVerse: Multimodal Memory for Lifelong Learning Agents](https://arxiv.org/abs/2512.03627) | [MemVerse](https://github.com/KnowledgeXLab/MemVerse) |
| NGM (Neural Graph Memory) | [Neural Graph Memory: A Structured Approach to Long-Term Memory in Multimodal Agents](https://www.researchgate.net/profile/Matt-Fisher-7/publication/394440420_Neural_Graph_Memory_A_Structured_Approach_to_Long-Term_Memory_in_Multimodal_Agents/links/689ab8c337b271210509c20f/Neural-Graph-Memory-A-Structured-Approach-to-Long-Term-Memory-in-Multimodal-Agents.pdf) | [Neural-Graph-Memory-NGM](https://github.com/StuckInTheNet/Neural-Graph-Memory-NGM) |



## ⚙️ Experimental Results

Our proposed M<sup>3</sup>Proctor framework achieves significant performance against state-of-the-art multimodal memory baselines.

<img src="figures/static.png">

Thanks to the proposed cost-aware cascade, M<sup>3</sup>Proctor escalates to raw visual sources only on demand, surpassing benchmarks on accuracy while controlling visual-token cost.

<img src="figures/dynamic.png">

## 📂 Project Structure

```
m3exam/
├── config/                 # unified YAML config + loader
├── execution/              # CLI entry points for each pipeline stage
│   ├── run_timeline.py     # core-event timeline generation
│   ├── run_questions.py    # question generation (8 types)
│   ├── run_finalize.py     # finalize persona into released layout
│   └── run_evaluation.py   # model / baseline evaluation
├── functions/              # pipeline implementations
│   ├── common/             # shared LLM / vision / JSON utilities
│   ├── timeline/           # timeline generation
│   ├── question_generation/# typed question generators
│   ├── thematic/           # thematic-subset construction
│   ├── finalize/           # dataset finalization
│   └── evaluation/         # scoring + multimodal-dependency analysis
├── m3proctor/              # M3Proctor multimodal memory method
├── example_set/            # one ready-to-inspect example persona
└── figures/                # paper figures
```

## 📄 License

This project is released under the [Apache License 2.0](LICENSE).

## 📝 Citation

```bibtex
TODO: citation to be added.
```

## 🙏 Acknowledgements

We acknowledge these excellent works for providing open-source code and inspiration: [nano-graphrag](https://github.com/gusye1234/nano-graphrag), [A-Mem](https://github.com/WujiangXu/A-mem), [Mem0](https://github.com/mem0ai/mem0), [MemoryOS](https://github.com/BAI-LAB/MemoryOS), [UniversalRAG](https://github.com/wgcyeo/UniversalRAG), [RAG-Anything](https://github.com/HKUDS/RAG-Anything), and [MIRIX](https://github.com/Mirix-AI/MIRIX).
