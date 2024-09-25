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
parser.add_argument("--model_weights", default=None, type=str, help="Path to model weights.")
parser.add_argument("--backbone", default="bert-large-uncased", type=str, help="Pre-trained transformer.")

class FileLoader:
    def __init__(self, directory):
        self._directory_path = Path(directory)
        self.sentence_to_document = {}
        self.documents = {}

        # 
        nlp = spacy.load("en_core_web_sm")
    
        # 
        for index, file_path in enumerate(self._directory_path.glob("*.txt")):
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                tokens = nlp(text)
                # Create dataset that assigns a list of sentences to each minute
                self.documents[index] = [sentence.text for sentence in tokens.sents]
    
                # Map from individual sentences back to individual minutes
                for sentence in tokens.sents:
                    self.sentence_to_document[sentence.text] = index


def main(args):
    # Load the model and its weights for inference
    backbone = transformers.AutoModel.from_pretrained(args.backbone)
    model = Model(args, backbone, CBMinutesDataset("../data"))
    model.load_state_dict(torch.load(args.model_weights))
    model.eval()

    # Load new data
    # TODO: Input is a dataframe with minutes individually split into their sentences
    # There must be a mapping from sentence to minutes to compute the overall sentiment of a minute
    # I assume that minutes to classify are saved as .txt files in data directory
    with open()

    # Preprocess new data
    tokenized_docs = tokenizer(documents, padding="longest", return_attention_mask=True) 

    # Classify new data


    # Save predicitions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)


