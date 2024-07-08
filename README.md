<h1 align="center">VISTA_Evaluation</h1>
<p align="center">
    <a href="https://arxiv.org/abs/2406.04292">
            <img alt="Build" src="http://img.shields.io/badge/cs.CV-arXiv%3A2406.04292-B31B1B.svg">
    </a>
    <a href="https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/visual">
        <img alt="Build" src="https://img.shields.io/badge/Github-VISTA Code-blue">
    </a>
    <a href="https://huggingface.co/BAAI/bge-visualized">
        <img alt="Build" src="https://img.shields.io/badge/🤗 Model-VISTA Model-yellow">
    </a>
    <a href="https://huggingface.co/datasets/JUNJIE99/VISTA_S2">
        <img alt="Build" src="https://img.shields.io/badge/🤗 Dataset-VISTA_S2 Dataset-yellow">
    </a>
</p>

This repository contains the evaluation code and datasets for reproducing the results presented in the ACL 2024 paper, [VISTA: Visualized Text Embedding For Universal Multi-Modal Retrieval](https://arxiv.org/abs/2406.04292). The original code of VISTA (also known as Visualized BGE) can be found [here](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/visual).

### Usage

Please follow the steps below to reproduce the results of Visualized-BGE-M3 on the WebQA dataset in the zero-shot evaluation setting:

1. Download the WebQA dataset [here](https://huggingface.co/datasets/JUNJIE99/VISTA_Evaluation).
> In our evaluation, we used all candidates from the webqa dataset to construct the corpus, including all candidates from the training and validation sets. Both pure text candidates and image-text candidates were included in the retrieval corpus. To ensure the validity of the results, we performed deduplication for text candidates, ensuring that each text candidate would not appear more than once in the corpus. For image-text candidates, we also performed deduplication. The criterion for deduplication was that the image ID and the corresponding text had to be identical. For instance, candidates with the same image ID but different texts were preserved as two distinct candidates.

2. Clone the repository from [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding), and place all files from the M3 directory in the ```./FlagEmbedding/Visual``` directory.

3. Configure the paths for the model weights, image directory and ```.jsonl``` files in ```eval_webqa.py```. Then, run ```eval_webqa.py```. The corresponding result in the paper is the Hybrid Corpus Recall@5.

We will continue to organize and upload more datasets and related code. If you have any questions or encounter any issues, please feel free to raise an issue.
