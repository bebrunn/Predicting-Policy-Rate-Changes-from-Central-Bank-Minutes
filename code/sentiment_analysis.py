import argparse
import datetime
import os
import re

import numpy as np
import torch
import torchmetrics
import transformers

from trainable_module import TrainableModule
from cbminutes_dataset import CBMinutesDataset

# Create argsparser to adjust arguments in shell.
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="../data", type=str, help="Data directory.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size used for training.")
parser.add_argument("--epochs", default=1, type=int, help="Number of training epochs.")
parser.add_argument("--seed", default=17, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--transformer", default="bert-base-uncased", type=str, help="Pre-trained transformer.")
# FIXME: Add more arguments if needed (e.g., dropout rate, size of dense layer, etc.).

# Create the Model class.
class Model(TrainableModule):
    def __init__(self, args, backbone, dataset):
        super().__init__()
        # Initialize Model class by defining it.
        self._backbone = backbone
        self._dense = torch.nn.Linear(backbone.config.hidden_size, backbone.config.hidden_size * 2)
        self._activation = torch.nn.ReLU()
        self._classifier = torch.nn.Linear(backbone.config.hidden_size * 2, len(dataset.label_vocab))

    # Implement the model computation.
    def forward(self, input_ids, attention_mask):
        hidden = self._backbone(input_ids, attention_mask).last_hidden_state
        hidden = self._dense(hidden)
        hidden = self._activation(hidden)
        hidden = self._classifier(hidden)
        # FIXME: Check output size and eventually use torch.mean() (?)
        return hidden
    

def main(args):
    # Set the random seed and number of threads.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # 
























dataset = CBMinutesDataset(directory="../data")

print(len(dataset.train.label_vocab))

