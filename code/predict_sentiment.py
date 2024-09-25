#!/usr/bin/env python3

import os
import argparse

import torch
import transformers
import pandas as pd
import spacy

# Import data set and model for inference
from cbminutes_dataset import CBMinutesDataset
from sentiment_analysis import Model

# Create argsparser to adjust arguments in shell.
parser = argparse.ArgumentParser()
parser.add_argument("--model_weights", default=None, type=str, help="Path of model weights.")
parser.add_argument("--backbone", default="bert-large-uncased", type=str, help="Pre-trained transformer.")

def main(args):
    # Load the model and its weights for inference
    backbone = transformers.AutoModel.from_pretrained(args.backbone)
    model = Model(args, backbone, CBMinutesDataset("../data"))
    model.load_state_dict(torch.load(args.model_weights))
    model.eval()

    # Load new data
    # TODO: Input is a dataframe with minutes individually split into their sentences
    # There must be a mapping from sentence to minutes to compute the overall sentiment of a minute



    # Preprocess new data
    tokenized_docs = tokenizer(documents, padding="longest", return_attention_mask=True) 

    # Classify new data


    # Save predicitions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)


