#!/usr/bin/env python3

import os
import argparse
from pathlib import Path

import torch
import transformers
import pandas as pd
import spacy
import numpy as np

# Import data set and model for inference
from cbminutes_dataset import CBMinutesDataset
from sentiment_analysis import parser, Model

# Create argsparser to adjust arguments in shell.
parser.add_argument("--model_weights", default=None, type=str, help="Path to model weights.")

class NewMinutes:
    def __init__(self, directory):
        self._directory_path = Path(directory)
        self.sentence_to_document = {}
        self.documents = {}

        # Load spacy object
        nlp = spacy.load("en_core_web_sm")
    
        # Extract files, split them in individual sentences, and create dictionaries that map documents <-> sentences.
        for file_path in self._directory_path.glob("*.txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                tokens = nlp(text)
                # Create dataset that assigns a list of sentences to each minute
                self.documents[file_path.name] = [sentence.text for sentence in tokens.sents]
    
                # Map from individual sentences back to individual minutes
                for sentence in tokens.sents:
                    self.sentence_to_document[sentence.text] = file_path.name


def main(args):
    # Load the model and its weights for inference
    backbone = transformers.AutoModel.from_pretrained(args.backbone)
    dataset = CBMinutesDataset("../data")
    model = Model(args, backbone, dataset)
    model.load_state_dict(torch.load(args.model_weights))
    model.eval()

    # Load new data
    minutes = NewMinutes("../data/new_data")
    documents = minutes.documents 

    # Initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.backbone)

    # Intialize dictionary to store overall
    document_sentiments = {}

    # Loop over documents and classify their sentences
    for doc_name, sentences in documents.items():

        # Tokenize the each sentence in a document
        tokenized_docs = tokenizer(sentences, padding="longest", return_attention_mask=True, return_tensors="pt")

        # Extract the input_ids and attention_masks from sentence tokenizations
        input_ids = tokenized_docs["input_ids"]
        attention_mask = tokenized_docs["attention_mask"]

        # Predict sentiment for each sentence
        with torch.no_grad():
            predictions = model(input_ids, attention_mask)
            sentence_predictions = torch.argmax(predictions, dim=1).cpu().numpy()

        # Determine sentiment of overall minute by majority vote
        overall_sentiment = np.bincount(sentence_predictions).argmax()

        # Store overall sentiment for each document
        document_sentiments[doc_name] = overall_sentiment

    # Save the predicition
    with open("sentiment_predictions.tsv", "w", encoding="utf-8") as file:
        for doc_name, sentiment in document_sentiments.items():
            file.write(f"{doc_name}\t{sentiment}\n")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
