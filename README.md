<h1 align="center"> 💫 M<sup>3</sup>Exam: Benchmarking Multimodal Memory for Realistic User-Agent Interactions </a></h2>

<div align="center">
    <a href="https://github.com/YuanchenBei/ColdRec/blob/main/LICENSE"><img src="https://badgen.net/github/license/YuanchenBei/ColdRec?color=green"></a>
    <a href="https://arxiv.org/abs/2601.03515">
    <img src="https://img.shields.io/badge/📃%20arXiv-Paper-b31b1b.svg"></a>
    <a href="https://huggingface.co/datasets/Ethan-Bei/Mem-Gallery">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow"></a>
</div>

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<h5 align="center">

<img src="figures/main.png">

</h5>

This is the official project repository for 

M<sup>3</sup>Exam is a novel query-centric multimodal conversational QA benchmark built on realistic user-agent interactions, enabling balanced multi-dimensional evaluation across multimodal memorizing, cross-modal reasoning, and implicit-intent interpreting over long-horizon histories of dialogue, images, and documents. We further propose M<sup>3</sup>Proctor, a modality-aware multimodal memory method that detects query modality bias and escalates to raw visual sources only on demand through a cost-aware cascade, enabling efficient multimodal evidence management with selective rather than indiscriminate visual injection.

## 🔧 Requirements

## 💫 Key Features

- **Accuracy**: Achieves state-of-the-art performance across diverse question types, including multi-hop and long-document QA tasks.
- **Efficiency**: Significantly reduces both graph construction time and token consumption compared to existing RAG baselines.
- **Incremental Updates**: Supports fast and efficient integration of new documents without requiring global tree reconstruction, enabling dynamic corpus adaptation.


## 🚀 Get Start

EraRAG and controled baselines are built on the unified framework proposed by [In-depth study of graphrag](https://github.com/JayLZhou/GraphRAG). Requirements.txt is included to help get you started. To run EraRAG, use the following command:
```
python main.py -opt <Method>.yaml -dataset_name <Datasetname> -external_tree <External tree path> -root <rootname> -query <wether to query>
```

On default, EraRAG will treat the input corpus as new corpus and enforce a global reconstruction. To make a insertion to a existing tree, set Dynamic.yaml key parameters as follows.

```
force: False
add: True
```


<!-- If you want to try reproducing the baseline methods, simply run:

```
bash run_baseline.sh
```

If you want to try reproducing the performance of IceBerg, simply run:

```
bash run_iceberg.sh
``` -->

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

Our proposed EraRAG framework achieves significant retrieval performance against state of the art graph-based RAG frameworks.

<img src="figures/static.png">

Thanks to the proposed selective reconstruction mechanism, EraRAG is able to perform fast insertions on evolving corpora, surpassing benchmarks on time and token cost reduction.

<img src="figures/dynamic.png">

## Acknowledgements

We acknowledge these excellent works for providing open-source code: [GraphRAG](https://github.com/microsoft/graphrag), [RAPTOR](https://github.com/parthsarthi03/raptor), [LightRAG](https://github.com/HKUDS/LightRAG), [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG), [In-depth study of graphrag](https://github.com/JayLZhou/GraphRAG).