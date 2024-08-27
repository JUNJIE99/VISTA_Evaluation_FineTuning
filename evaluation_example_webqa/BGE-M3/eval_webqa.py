import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
print(os.getcwd())
sys.path.append('./FlagEmbedding/visual')

import faiss
import torch
import logging
import datasets
import numpy as np
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser
# from FlagEmbedding import FlagModel
from flag_eva_m3 import Flag_bgev_model
# from flag_clip import Flag_clip
import json

logger = logging.getLogger(__name__)


@dataclass
class Args:
    resume_path: str = field(
        default="your_model_weight_path", 
        metadata={'help': 'The model checkpoint path.'}
    )
    image_dir: str = field(
        default="your_image_directory_path",
        metadata={'help': 'Where the images located on.'}
    )
    encoder: str = field(
        default="BAAI/bge-m3",
        metadata={'help': 'The encoder name or path.'}
    )
    fp16: bool = field(
        default=False,
        metadata={'help': 'Use fp16 in inference?'}
    )
    add_instruction: bool = field(
        default=False,
        metadata={'help': 'Add query-side instruction?'}
    )
    
    max_query_length: int = field(
        default=512,
        metadata={'help': 'Max query length.'}
    )
    max_passage_length: int = field(
        default=512, 
        metadata={'help': 'Max passage length.'}
    )
    batch_size: int = field(
        default=256,
        metadata={'help': 'Inference batch size.'}
    )
    index_factory: str = field(
        default="Flat",
        metadata={'help': 'Faiss index factory.'}
    )
    k: int = field(
        default=100,
        metadata={'help': 'How many neighbors to retrieve?'}
    )

    save_embedding: bool = field(
        default=False,
        metadata={'help': 'Save embeddings in memmap at save_dir?'}
    )
    load_embedding: bool = field(
        default=False,
        metadata={'help': 'Load embeddings from save_dir?'}
    )
    save_path: str = field(
        default="embeddings.memmap",
        metadata={'help': 'Path to save embeddings.'}
    )



def index(model: Flag_bgev_model, corpus: [datasets.Dataset, datasets.Dataset], batch_size: int = 256, max_length: int=512, index_factory: str = "Flat", save_path: str = None, save_embedding: bool = False, load_embedding: bool = False):
    """
    1. Encode the entire corpus into dense embeddings; 
    2. Create faiss index; 
    3. Optionally save embeddings.
    """
    if load_embedding:
        test = model.encode("test")
        dtype = test.dtype
        dim = len(test)

        corpus_embeddings = np.memmap(
            save_path,
            mode="r",
            dtype=dtype
        ).reshape(-1, dim)
    
    else:
        text_corpus = corpus[0]
        mm_it_corpus = corpus[1]
        
        text_corpus_embeddings = model.encode_corpus(text_corpus, batch_size=batch_size, max_length=max_length, corpus_type='text')
        mm_it_corpus_embeddings = model.encode_corpus(mm_it_corpus, batch_size=batch_size, max_length=max_length, corpus_type='mm_it')

        
        corpus_embeddings = np.concatenate([text_corpus_embeddings, mm_it_corpus_embeddings], axis=0)
        
        dim = corpus_embeddings.shape[-1]
        
        if save_embedding:
            logger.info(f"saving embeddings at {save_path}...")
            memmap = np.memmap(
                save_path,
                shape=corpus_embeddings.shape,
                mode="w+",
                dtype=corpus_embeddings.dtype
            )

            length = corpus_embeddings.shape[0]
            # add in batch
            save_batch_size = 10000
            if length > save_batch_size:
                for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
                    j = min(i + save_batch_size, length)
                    memmap[i: j] = corpus_embeddings[i: j]
            else:
                memmap[:] = corpus_embeddings
    
    # create faiss index
    faiss_index_all = faiss.index_factory(dim, index_factory, faiss.METRIC_INNER_PRODUCT)
    faiss_index_text = faiss.index_factory(dim, index_factory, faiss.METRIC_INNER_PRODUCT)
    faiss_index_mm_it = faiss.index_factory(dim, index_factory, faiss.METRIC_INNER_PRODUCT)

    if model.device == torch.device("cuda"):
        # co = faiss.GpuClonerOptions()
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True

        faiss_index_all = faiss.index_cpu_to_all_gpus(faiss_index_all, co)
        faiss_index_text = faiss.index_cpu_to_all_gpus(faiss_index_text, co)
        faiss_index_mm_it = faiss.index_cpu_to_all_gpus(faiss_index_mm_it, co)

    # NOTE: faiss only accepts float32
    logger.info("Adding embeddings...")
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    faiss_index_all.train(corpus_embeddings)
    faiss_index_all.add(corpus_embeddings)

    text_corpus_embeddings = text_corpus_embeddings.astype(np.float32)
    faiss_index_text.train(text_corpus_embeddings)
    faiss_index_text.add(text_corpus_embeddings)

    mm_it_corpus_embeddings = mm_it_corpus_embeddings.astype(np.float32)
    faiss_index_mm_it.train(mm_it_corpus_embeddings)
    faiss_index_mm_it.add(mm_it_corpus_embeddings)

    return faiss_index_all, faiss_index_text, faiss_index_mm_it


def search(model: Flag_bgev_model, queries: datasets, faiss_index: faiss.Index, k:int = 100, batch_size: int = 256, max_length: int=512):
    """
    1. Encode queries into dense embeddings;
    2. Search through faiss index
    """
    query_embeddings = model.encode_queries(queries["q_text"], batch_size=batch_size, max_length=max_length, query_type="text")
    query_size = len(query_embeddings)
    
    all_scores = []
    all_indices = []
    
    for i in tqdm(range(0, query_size, batch_size), desc="Searching"):
        j = min(i + batch_size, query_size)
        query_embedding = query_embeddings[i: j]
        score, indice = faiss_index.search(query_embedding.astype(np.float32), k=k)
        all_scores.append(score)
        all_indices.append(indice)
    
    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    return all_scores, all_indices
    
    
def evaluate(preds, labels, cutoffs=[1,5,10,20,50,100]):
    """
    Evaluate MRR and Recall at cutoffs.
    """
    metrics = {}
    
    # MRR
    mrrs = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        jump = False
        for i, x in enumerate(pred, 1):
            if x in label:
                for k, cutoff in enumerate(cutoffs):
                    if i <= cutoff:
                        mrrs[k] += 1 / i
                jump = True
            if jump:
                break
    mrrs /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        mrr = mrrs[i]
        metrics[f"MRR@{cutoff}"] = mrr

    # Recall
    recalls = np.zeros(len(cutoffs))
    easy_recalls = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        if not isinstance(label, list):
            label = [label]
        for k, cutoff in enumerate(cutoffs):
            recall = np.intersect1d(label, pred[:cutoff])
            recalls[k] += len(recall) / len(label)
            if len(recall) > 0:
                easy_recalls[k] += 1
    recalls /= len(preds)
    easy_recalls /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        recall = recalls[i]
        metrics[f"Recall@{cutoff}"] = recall
    
    for i, cutoff in enumerate(cutoffs):
        easy_recall = easy_recalls[i]
        metrics[f"Easy_Recall@{cutoff}"] = easy_recall

    return metrics
    

def main():
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]

    eval_data = datasets.load_dataset('json', data_files="the_path_to/val_query.jsonl", split='train')
    mm_it_corpus = datasets.load_dataset('json',  data_files="the_path_to/mm_it_corpus.jsonl", split='train')
    text_corpus = datasets.load_dataset('json', data_files="the_path_to/text_corpus.jsonl", split='train')
    
    model = Flag_bgev_model(model_name_bge = "BAAI/bge-m3",
                        model_name_eva = "EVA02-CLIP-L-14", # "EVA02-CLIP-B-16",
                        normlized = True,
                        eva_pretrained_path = "eva_clip",
                        resume_path=args.resume_path,
                        image_dir=args.image_dir,
                        )
    
    print(args.resume_path)
    
    faiss_index_all, faiss_index_text, faiss_index_mm_it = index(
        model=model, 
        corpus=[text_corpus, mm_it_corpus], 
        batch_size=args.batch_size,
        max_length=args.max_passage_length,
        index_factory=args.index_factory,
        save_path=args.save_path,
        save_embedding=args.save_embedding,
        load_embedding=args.load_embedding
    )

    scores, indices = search(
        model=model, 
        queries=eval_data, 
        faiss_index=faiss_index_all, 
        k=args.k, 
        batch_size=args.batch_size, 
        max_length=args.max_query_length
    )
    
    ### merge corpus
    dataset_dict = datasets.DatasetDict({"text": text_corpus, "mm_it": mm_it_corpus})
    all_corpus = datasets.concatenate_datasets([dataset_dict["text"], dataset_dict["mm_it"]])

    retrieval_results = []
    for indice in indices:
        # filter invalid indices
        indice = indice[indice != -1].tolist()
        retrieval_results.append(all_corpus[indice]["content"])

    ground_truths = []
    for sample in eval_data:
        ground_truths.append(sample["positive"])
        
    metrics = evaluate(retrieval_results, ground_truths)
    print("Hybrid Corpus:")
    print(metrics)


    text_queries = eval_data.filter(lambda sample: sample['type'] == "text")
    mm_it_queries = eval_data.filter(lambda sample: sample['type'] == "mm_it")
    
    scores, text_indices = search(
        model=model, 
        queries=text_queries, 
        faiss_index=faiss_index_text, 
        k=args.k, 
        batch_size=args.batch_size, 
        max_length=args.max_query_length
    )
    retrieval_results = []
    for indice in text_indices:
        # filter invalid indices
        indice = indice[indice != -1].tolist()
        retrieval_results.append(text_corpus[indice]["content"])

    ground_truths = []
    for sample in text_queries:
        ground_truths.append(sample["positive"])
        
    metrics = evaluate(retrieval_results, ground_truths)
    print("text tasks:")
    print(metrics)


    scores, mm_it_indices = search(
        model=model, 
        queries=mm_it_queries, 
        faiss_index=faiss_index_mm_it, 
        k=args.k, 
        batch_size=args.batch_size, 
        max_length=args.max_query_length
    )
    retrieval_results = []
    for indice in mm_it_indices:
        # filter invalid indices
        indice = indice[indice != -1].tolist()
        retrieval_results.append(mm_it_corpus[indice]["content"])

    ground_truths = []
    for sample in mm_it_queries:
        ground_truths.append(sample["positive"])
        
    metrics = evaluate(retrieval_results, ground_truths)
    print("mm_it tasks:")
    print(metrics)


if __name__ == "__main__":
    main()