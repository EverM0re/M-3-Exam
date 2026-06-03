<h1 align="center"> 💫 M<sup>3</sup>Exam: Benchmarking Multimodal Memory for Realistic User-Agent Interactions </a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<h5 align="center">

<img src="figures/main.png">

</h5>

EraRAG is a novel hierarchical graph construction framework that supports dynamic updates through localized selective re-partitioning, enabling efficient and scalable retrieval with strong static accuracy and stable performance under corpus changes.

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



## ⚙️ Experimental Results

Our proposed EraRAG framework achieves significant retrieval performance against state of the art graph-based RAG frameworks.

<img src="figures/static.png">

Thanks to the proposed selective reconstruction mechanism, EraRAG is able to perform fast insertions on evolving corpora, surpassing benchmarks on time and token cost reduction.

<img src="figures/dynamic.png">

## Acknowledgements

We acknowledge these excellent works for providing open-source code: [GraphRAG](https://github.com/microsoft/graphrag), [RAPTOR](https://github.com/parthsarthi03/raptor), [LightRAG](https://github.com/HKUDS/LightRAG), [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG), [In-depth study of graphrag](https://github.com/JayLZhou/GraphRAG).