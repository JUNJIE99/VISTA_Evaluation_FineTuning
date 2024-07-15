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
</p>

<p align="center">
</a>
    <a href="https://huggingface.co/datasets/JUNJIE99/VISTA_S2">
        <img alt="Build" src="https://img.shields.io/badge/🤗 Dataset-VISTA S2 Training Dataset-yellow">
    </a>
    <a href="https://huggingface.co/datasets/JUNJIE99/VISTA_Evaluation">
        <img alt="Build" src="https://img.shields.io/badge/🤗 Dataset-Zero_Shot Multimodal Retrieval Dataset-yellow">
    </a>
</p>

This repository contains the evaluation code and datasets for reproducing the results presented in the ACL 2024 paper, [VISTA: Visualized Text Embedding For Universal Multi-Modal Retrieval](https://arxiv.org/abs/2406.04292). The original code of VISTA (also known as Visualized BGE) can be found in [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/visual).

### Usage

Please follow the steps below to reproduce the results of Visualized-BGE-M3 on the WebQA dataset in the zero-shot evaluation setting:

1. Download the WebQA dataset [here](https://huggingface.co/datasets/JUNJIE99/VISTA_Evaluation).
> In our evaluation process, we built our corpus utilizing all candidates from the WebQA dataset, which encompassed both the training and validation sets. This corpus included both text-only and image-text candidates. To ensure the accuracy of the results, we implemented deduplication across all candidates. For text-only candidates, we made certain that each unique piece of text was represented just once within the corpus. In the case of image-text candidates, we also carried out deduplication. The criterion for this was that both the image ID and the associated text had to be identical. Hence, candidates sharing the same image ID but with differing texts were maintained as distinct candidates.

2. Clone the repository from [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding), and place all files from the [webqa/BGE_M3](https://github.com/JUNJIE99/VISTA_Evaluation/tree/main/webqa/BGE-M3) directory in the ```./FlagEmbedding/Visual``` directory.

3. Configure the paths for the model weights, image directory and ```.jsonl``` files in ```eval_webqa.py```. Then, run ```eval_webqa.py```. The corresponding result in the paper is the Hybrid Corpus Recall@5.

We will continue to organize and upload more datasets and related code. If you have any questions or encounter any issues, please feel free to raise an issue.
