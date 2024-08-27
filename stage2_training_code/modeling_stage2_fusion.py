import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoModel
from transformers.file_utils import ModelOutput

import os

import sys
sys.path.append("./EVA-CLIP/rei")
from eva_clip import create_eva_vision_and_transforms, get_tokenizer

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    c_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class mlp(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, 4 * input_dim)
        self.fc2 = nn.Linear(4 * input_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, output_dim)

        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x

class BGE_EVAToken(nn.Module):
    '''
    BGE + CLIP-V
    '''
    
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 model_name_bge: str = None,
                 model_name_eva: str = "EVA02-CLIP-B-16",
                 normlized: bool = True, #False,
                 sentence_pooling_method: str = 'cls',
                 negatives_cross_device: bool = False,
                 temperature: float = 0.02, # 1.0
                 eva_pretrained_path = None,
                 writer:SummaryWriter = None
                 ):
        super().__init__()

        if eva_pretrained_path is None:
            eva_pretrained_path = "eva_clip" # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"


        self.bge_encoder = AutoModel.from_pretrained(model_name_bge).encoder
        self.bge_embeddings = AutoModel.from_pretrained(model_name_bge).embeddings
        self.bge_pooler = AutoModel.from_pretrained(model_name_bge).pooler

        self.model_visual, self.preprocess_train, self.preprocess_val= create_eva_vision_and_transforms(
            model_name_eva, 
            eva_pretrained_path, 
            force_custom_clip=True)

        self.visual_proj = nn.Linear(768, 768)

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        if not normlized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            #     logger.info("Run in a single GPU, set negatives_cross_device=False")
            #     self.negatives_cross_device = False
            # else:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        self.tf_writer = writer

    def gradient_checkpointing_enable(self, **kwargs):
        # self.bge_encoder.gradient_checkpointing_enable()
        self.model_visual.set_grad_checkpointing(True)
    
    def get_extended_attention_mask(
        self, attention_mask: Tensor, input_shape: Tuple[int], device: torch.device = None, dtype: torch.float = torch.float16
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        
        return extended_attention_mask

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]

    
    
    def t_encoder(self, texts):
        '''
        encode captions only, use for training
        '''
        input_ids = texts['input_ids']
        attention_mask = texts['attention_mask']

        input_shape = input_ids.size()
        device = input_ids.device

        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        head_mask = [None] * 12
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        
        embedding_output = self.bge_embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
            past_key_values_length=0,
        )
        encoder_outputs = self.bge_encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.bge_pooler(sequence_output) if self.bge_pooler is not None else None

        t_reps = self.sentence_embedding(sequence_output, texts['attention_mask']) # tensor: reps with pooling
        if self.normlized:
            t_reps = torch.nn.functional.normalize(t_reps, dim=-1)
        return t_reps.contiguous()

    def mm_encoder(self, images, prompts):
        img_token_emb = self.img_token_embedding(images) #[B, T, C]
        if img_token_emb is not None:
            img_token_emb = img_token_emb[:,1:]
            img_token_emb = self.visual_proj(img_token_emb)
            device = img_token_emb.device
            
            img_token_len = img_token_emb.size()[1]

            # image position embedding
            img_token_position_ids = torch.arange(1, 1 + img_token_len).to(device=device)
            img_position_embeddings = self.bge_embeddings.position_embeddings(img_token_position_ids)
            img_token_emb = img_token_emb + img_position_embeddings

            img_token_emb = self.bge_embeddings.LayerNorm(img_token_emb)
        else:
            return self.t_encoder(prompts)

        prompt_input_ids = prompts['input_ids']
        prompt_attention_mask = prompts['attention_mask']
        prom_input_shape = prompt_input_ids.size()
        
        batch_size = prom_input_shape[0]
        prompt_len = prom_input_shape[1]
        prompt_start = 1 + img_token_len

        
        
        cls_id = torch.tensor([0]).to(device=device)
        prompt_position_ids = torch.arange(prompt_start, prompt_start + prompt_len - 1).to(device=device)
        prompt_position_ids = torch.cat([cls_id, prompt_position_ids]).to(device=device)

        prompt_token_type_ids = torch.zeros(prom_input_shape, dtype=torch.long, device=device)
        prompt_embedding_output = self.bge_embeddings(
            input_ids=prompt_input_ids,
            position_ids=prompt_position_ids,
            token_type_ids=prompt_token_type_ids,
            inputs_embeds=None,
            past_key_values_length=0,
        )  # [B, T, C]

        ### prompt+img embeddings --> encoder
        cls_token = prompt_embedding_output[:, 0:1, :]
        prompt_embedding_output = prompt_embedding_output[:, 1:]

        prompt_img_embedding = torch.cat([cls_token, img_token_emb, prompt_embedding_output], dim=1) 
        
        img_attention_mask = torch.ones(batch_size, img_token_len, device=device)  
        prom_img_attention_mask = torch.cat([img_attention_mask, prompt_attention_mask], dim=1)
        prom_img_input_shape = prompt_img_embedding.size()

        head_mask = [None] * 12
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(prom_img_attention_mask, prom_img_input_shape)
        
        encoder_outputs = self.bge_encoder(
            prompt_img_embedding,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        sequence_output = encoder_outputs[0]
        
        prompt_img_reps = self.sentence_embedding(sequence_output, prom_img_attention_mask) # tensor: reps with pooling
        if self.normlized:
            prompt_img_reps = torch.nn.functional.normalize(prompt_img_reps, dim=-1)
        return prompt_img_reps

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def img_token_embedding(self, images):
        '''
        Used for training
        '''
        if images is None:
            return None
        img_token_emb = self.model_visual.encode_image(images, normalize=False) #return_all_features=True, [B, T, 768] 
        
        return img_token_emb.contiguous()
    
    def encode_text(self, texts):
        '''
        now, this function is used for CLIP_Benchmarking
        return pooling + linear adaptor features of texts but without normalization
        '''
        
        if texts is None:
            return None
        
        return self.t_encoder(texts)
    
    def encode_image_only(self, images, prompts = None):
        '''
        now, this function is used for Retrival Benchmarking
        the difference with encode_image is no prompts, just a bge cls
        '''
        if images is None:
            return None
        
        if prompts is None:
            prompts = {'input_ids': torch.tensor([101, 102]),  
          'attention_mask': torch.tensor([1,1])}
            device = images.device
            batch_size = images.shape[0]
            prompt_input_ids = prompts['input_ids'].to(device).unsqueeze(0).repeat(batch_size, 1)
            prompt_attention_mask = prompts['attention_mask'].to(device).unsqueeze(0).repeat(batch_size, 1)
            prompts = {'input_ids': prompt_input_ids,  
                       'attention_mask': prompt_attention_mask}
        
        img_reps = self.mm_encoder(images, prompts)
        return img_reps
        

    def forward(self, mm_it_query=None, image_candidate=None, text_candidate=None, text_query=None, mm_it_candidate=None, task_type=None):
        ### for stage-2 training
        if task_type == "edit_image":
            mm_query_reps = self.mm_encoder(mm_it_query[0], mm_it_query[1])
            image_candi_reps = self.encode_image_only(image_candidate) 
            query_reps = mm_query_reps
            candi_reps = image_candi_reps
        elif task_type == "t2it":
            text_query_reps = self.encode_text(text_query)
            mmit_candi_reps = self.mm_encoder(mm_it_candidate[0], mm_it_candidate[1])
            query_reps = text_query_reps
            candi_reps = mmit_candi_reps
            
        
        if self.training:
            if self.negatives_cross_device:
                query_reps = self._dist_gather_tensor(query_reps)
                candi_reps = self._dist_gather_tensor(candi_reps)

            scores = self.compute_similarity(query_reps, candi_reps)
            scores = scores / self.temperature
            scores = scores.view(query_reps.size(0), -1)
            
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (candi_reps.size(0) // query_reps.size(0))
            
            loss_edit = self.compute_loss(scores, target)
            loss = loss_edit

            self.tf_writer.add_scalar("loss_image_edit", loss_edit)
            
            logging.info("task types: %s; loss: %s" %(task_type, str(loss_edit)))
        else:
            scores = self.compute_similarity(query_reps, candi_reps)
            loss=None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=query_reps,
            c_reps=candi_reps,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        torch.save(self.state_dict(), os.path.join(output_dir, 'BGE_EVA_Token.pth'))