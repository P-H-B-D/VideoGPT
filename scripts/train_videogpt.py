import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="6, 7"
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from videogpt import VideoGPT, VideoData
# your favorite machine learning tracking tool
from pytorch_lightning.loggers import WandbLogger

import torch

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to train on')
    parser = VideoGPT.add_model_specific_args(parser)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--max_steps', type=int, default=None)
    args = parser.parse_args()

    # Initialize data handling
    data = VideoData(args)
    # Ensure that the data loaders are initialized (may involve caching/preprocessing)
    data.train_dataloader()
    data.test_dataloader()

    # Set class conditional dimension if needed
    args.class_cond_dim = data.n_classes if hasattr(args, 'class_cond') and args.class_cond else None
    
    # Initialize the model
    model = VideoGPT(args)

    # Set up the training callbacks
    callbacks = [ModelCheckpoint(monitor='val/loss', mode='min', save_top_k=-1)]
    
    # Prepare the Trainer configuration
    trainer_kwargs = {
        'callbacks': callbacks,
        'max_steps': args.max_steps,
        'devices': args.gpus if args.gpus > 0 else None,  # Support CPU training if gpus is 0
        'accelerator': 'gpu' if args.gpus > 0 else None,
    }
    if args.gpus > 1:
        # Adjust training strategy for distributed GPU training
        trainer_kwargs['strategy'] = 'ddp_find_unused_parameters_false'
    
    # Initialize the Trainer
    trainer = pl.Trainer(**trainer_kwargs)

    # Start training
    trainer.fit(model, data)

if __name__ == "__main__":
    main()

