import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2, 3, 4, 5, 6, 7"
os.environ["WANDB_MODE"] = "disabled"
import logging
import os
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl
)
import sys
print(os.getcwd())
sys.path.append('./FlagEmbedding/baai_general_embedding/downstream')
from arguments import ModelArguments, DataArguments, \
    RetrieverTrainingArguments as TrainingArguments
from data_ds_cirr import Multimodal_Dataset, Multimodal_Collator
from modeling_ds_cirr import BGE_EVAToken
from trainer import BiTrainer
import torch
from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)

from dataclasses import dataclass


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)



    num_labels = 1
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.bge_model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.bge_model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    logger.info('Config: %s', config)
    
    writer = SummaryWriter(os.path.join(training_args.output_dir, "my_tfwriter"))
    model = BGE_EVAToken(model_name_bge = model_args.bge_model_name_or_path,
                                model_name_eva = model_args.visual_model_name_or_path, # "EVA02-CLIP-B-16",
                                normlized = training_args.normlized,
                                sentence_pooling_method = training_args.sentence_pooling_method,
                                negatives_cross_device = training_args.negatives_cross_device,
                                temperature = training_args.temperature,
                                writer=writer,
                                eva_pretrained_path = "eva_clip",)

    
    
    if training_args.resume_path is None:
        logger.info('Training from scratch')
    else:
        logger.info('Traing from checkpoint: %s', training_args.resume_path)
        model.load_state_dict(torch.load(training_args.resume_path, map_location='cpu'))

    if training_args.fix_position_embedding:
        for k, v in model.named_parameters():
            if "position_embeddings" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False
    

    if not model_args.train_vision_tower:
        if model_args.custom_train_vision_tower is not None:
            logger.info('You can not require not training vision tower but ask to train specific layers!') 
            return
        for k, v in model.named_parameters():
            if "model_clipv" in k or "model_visual" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False
    else:
        if model_args.custom_train_vision_tower is None:
            pass
        else:
            train_num = model_args.custom_train_vision_tower
            freeze_num = 12 - train_num
            freeze_layers = []
            for _i in range(freeze_num):
                layer_name = "model_visual.visual.blocks." + str(_i) + "."
                freeze_layers.append(layer_name)
            for k, v in model.named_parameters():
                if any(layer_name in k for layer_name in freeze_layers):
                    logging.info(f"Freeze the parameters for {k}")
                    v.requires_grad = False 

    if not model_args.train_text_tower:
        for k, v in model.named_parameters():
            if "bge_encoder" in k or "bge_embeddings" in k or "bge_pooler" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False 


    train_dataset = Multimodal_Dataset(args=data_args, 
                                       image_processor=model.preprocess_train,
                                      )

    trainer = BiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=Multimodal_Collator(
            tokenizer,
        ),
        tokenizer=tokenizer,
    )
 
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    trainer.train()
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
    