### Support for Alternating Batch Training of Different Tasks
import os.path
import random

import datasets
from torch.utils.data import Dataset, IterableDataset

from PIL import Image
import json
import torch
import torch.distributed
from arguments import DataArguments

import math
import numpy as np

class Multimodal_Dataset(Dataset):
    def __init__(self, args:DataArguments, batch_size, seed=123, image_processor=None, process_index = 0, num_processes=1) -> None:
        self.image_dir = args.train_data_image
        self.train_group_size = args.train_group_size
        self.t2it_size = 2 # self.train_group_size
        
        jsonl_dir = args.train_data
        
        edit_image_anno = os.path.join(jsonl_dir, "Image_Edit_307K_filter.jsonl")# "instructpix2pix_train_filter.jsonl"
        t2it_anno = os.path.join(jsonl_dir, "T2IT.jsonl")
        
        self.edit_dataset = datasets.load_dataset('json', data_files=edit_image_anno, split='train')
        self.t2it_dataset = datasets.load_dataset('json', data_files=t2it_anno, split='train')
        
        self.naive_t2it_dataset = datasets.load_dataset('json', data_files=naive_t2it_anno, split='train')
        
        #### unify all key-value data types
        self.edit_dataset = self.edit_dataset.map(self.positive2list)
        self.edit_dataset = self.edit_dataset.map(self.hard2lists)
        self.t2it_dataset = self.t2it_dataset.map(self.query2list)
        
        
        trainset_list = [self.edit_dataset, self.t2it_dataset]
        
        self.each_data_indx = []
        start_num = 0
        
        
        for set in trainset_list:
            _each_data_indx = np.arange(start_num, start_num + len(set))
            self.each_data_indx.append(_each_data_indx)
            start_num = start_num + len(set)
        
        self.dataset = datasets.concatenate_datasets(trainset_list)
        
        self.image_processor = image_processor
        
        self.process_index = process_index
        self.batch_size = batch_size # per_device batch size
        self.num_processes = num_processes
        self.overall_batch_size = batch_size * num_processes
        
        self.args = args
        
        self.deterministic_generator = np.random.default_rng(seed)
        self.step = 0
        self.refresh_epoch()

    def query2list(self, data):
        data["query"] = [data["query"]]
        return data
    def positive2list(self, data):
        data["positive"] = [data["positive"]]
        return data
    
    def hard2lists(self, data):
        data["hard_negative"] = [[hn] for hn in data["hard_negative"]]
        return data
    
    def refresh_epoch(self):
        print(f'---------------------------*Rank {self.process_index}: refresh data---------------------------')

        batch_datas = []
        
        for data_indx in self.each_data_indx: #[[0,1,2...], [N+0, N+1, N+2,...], ...]
            self.deterministic_generator.shuffle(data_indx)
            iter_num = len(data_indx) // self.overall_batch_size
            for i in range(iter_num):
                _indx = np.arange(i*self.overall_batch_size, (i+1)*self.overall_batch_size)
                batch_datas.append(data_indx[_indx])
        
        self.deterministic_generator.shuffle(batch_datas)
        
        self.batch_datas = batch_datas
        self.steps = 0
    
    def __getitem__(self, idx):
        batch_indx = self.batch_datas[self.steps]
        cur_batch_size = int(len(batch_indx) / self.num_processes)
        assert cur_batch_size == self.batch_size
        
        batch_indx = batch_indx[self.process_index * cur_batch_size: (self.process_index + 1) * cur_batch_size]
        
        batch_data = self.dataset[batch_indx]
        self.steps += 1
        
        return self.create_batch_data(batch_raw_data=batch_data)
        
        
    def create_batch_data(self, batch_raw_data):
        q_type = batch_raw_data["q_type"][0]
        c_type = batch_raw_data["c_type"][0]
        queries, candidates, dataset_type = [], [], []
        if q_type == "mm_it" and c_type == "image":
            ## edit_images
            for i in range(len(batch_raw_data["query"])):
                
                q_image, q_text = batch_raw_data["query"][i]
                p_image = batch_raw_data["positive"][i][0]
                if len(batch_raw_data["hard_negative"][i]) >= self.train_group_size - 1:
                    _hn_images = random.sample(batch_raw_data["hard_negative"][i], self.train_group_size - 1)
                else:
                    _hn_images = random.choices(batch_raw_data["hard_negative"][i], k = self.train_group_size - 1) 
                _hn_images = [_hn[0] for _hn in _hn_images]  
                
                q_image = self.image_processor(self.img2pil(q_image))
                
                query = (q_image, q_text)
                
                candidate_images = []
                p_image = self.image_processor(self.img2pil(p_image))
                candidate_images.append(p_image)
                
                hn_images = []
                for _i in _hn_images:
                    _i = self.image_processor(self.img2pil(_i))
                    hn_images.append(_i)
                candidate_images.extend(hn_images)
                
                queries.append(query)
                candidates.append(candidate_images)
                dataset_type.append("edit_image")
                
        elif q_type == "text" and c_type == "mm_it":
            # t2it
            for i in range(len(batch_raw_data["query"])):
                q_text = batch_raw_data["query"][i][0]
                p_image, p_text = batch_raw_data["positive"][i]
                _hn_mm_it = random.sample(batch_raw_data["hard_negative"][i], self.t2it_size - 1)
                
                p_image = self.image_processor(self.img2pil(p_image))
                
                _hn_mm_it = [(self.image_processor(self.img2pil(_hn[0])), _hn[1]) for _hn in _hn_mm_it]
                
                candidate_mm_it = []
                candidate_mm_it.append((p_image, p_text))
                
                candidate_mm_it.extend(_hn_mm_it)
                
                queries.append(q_text)
                candidates.append(candidate_mm_it)
                dataset_type.append("t2it")
            
        else:
            raise NotImplementedError("Unsupported dataset, please check data format.")
        
        
        return queries, candidates, dataset_type
        
    def __len__(self):
        return len(self.batch_datas) * self.num_processes
    
    def img2pil(self, image_path):
        complelte_img_path = os.path.join(self.image_dir, image_path)
        return Image.open(complelte_img_path)
    
    
    


class Multimodal_Collator:
    def __init__(self, tokenizer, mmit_max_len=109, pure_text_max_len=512):
        self.tokenizer = tokenizer
        self.query_max_len = mmit_max_len
    
    def reshape_image_candidate(self, i_candidates):
        all_candidates = []
        for group in i_candidates:
            for image in group:
                all_candidates.append(image)
        return all_candidates
    
    def reshape_text_candidate(self, t_candidates):
        all_candidates = []
        for group in t_candidates:
            for text in group:
                all_candidates.append(text)
        return all_candidates
    
    def reshape_mmit_candidate(self, mm_candidates):
        all_candidates = []
        for group in mm_candidates:
            for mm in group:
                all_candidates.append(mm)
        return all_candidates
    
    
    def __call__(self, features):
        
        features = features[0] 
        task_type = features[2][0]
        
        if task_type == "edit_image":
            ### edit image
            mm_it_query = features[0]
            image_candidate = features[1]
            
            q_images = [f[0] for f in mm_it_query]
            q_texts = [f[1] for f in mm_it_query]
            
            q_text_collated = self.tokenizer(
                q_texts,
                padding= True, #"max_length",
                truncation=True,
                max_length=self.query_max_len,
                return_tensors="pt",
            )
            q_image_collated = torch.stack(q_images)
            
            
            c_images = self.reshape_image_candidate(image_candidate)
            c_image_collated = torch.stack(c_images)

            return {"mm_it_query": (q_image_collated, q_text_collated), "image_candidate": c_image_collated, "task_type": "edit_image"}
        
        elif task_type == "t2it":
            #### t2it
            text_query = features[0]
            mmit_candidate = features[1]
            
            mmit_candidate = self.reshape_mmit_candidate(mmit_candidate)
            
            c_images = [f[0] for f in mmit_candidate]
            c_texts = [f[1] for f in mmit_candidate]
            
            
            c_text_collated = self.tokenizer(
                c_texts,
                padding= True, #"max_length",
                truncation=True,
                max_length=self.query_max_len,
                return_tensors="pt",
            )
            c_image_collated = torch.stack(c_images)
            
            q_text_collated = self.tokenizer(
                text_query,
                padding= True, #"max_length",
                truncation=True,
                max_length=self.query_max_len,
                return_tensors="pt",
            )
    
            return {"text_query": q_text_collated, "mm_it_candidate": (c_image_collated, c_text_collated), "task_type": "t2it"}
        else:
            raise NotImplementedError
        
        
        