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
from cbminutes_dataset import Vocabulary, CBMinutesDataset
from sentiment_analysis import parser, Model

# Create argsparser to adjust arguments in shell.
parser.add_argument("--model_weights", default=None, type=str, help="Path to model weights.")

class UnseenMinutes:
    # Create dataset class of NewMinutes.
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, sentences, tokenizer):
            self._sentences = sentences
            self._tokenizer = tokenizer

        def  __len__(self):
            return len(self._sentences)

        def __getitem__(self, index):
            return self._sentences[index]

        def tokenize(self, batch):
            return self._tokenizer(
                batch, padding="longest", return_attention_mask=True, return_tensors="pt"
            )

    def __init__(self, directory, nlp):
        self._directory_path = Path(directory)
        self.sentence_to_document = {}
        self.documents = {}

        # Create lists of files and their texts
        files = list(self._directory_path.glob("*.txt"))
        files.sort(key=lambda x: x.name)
        docs = [open(file_path, "r", encoding="utf-8").read() for file_path in files]

        for doc, file_path in zip(nlp.pipe(docs), files):
            # Create dataset that assigns a list of sentences to each minute
            self.documents[file_path.name] = [sentence.text.strip() for sentence in doc.sents]


def main(args):
    # Load the model and its weights for inference
    backbone = transformers.AutoModel.from_pretrained(args.backbone)
    dataset = CBMinutesDataset("../data")
    model = Model(args, backbone, dataset.train)
    model.load_state_dict(torch.load(args.model_weights))
    model.eval()

    # Move model to GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load spacy NLP pipeline for sentence tokenization
    nlp = spacy.load("en_core_web_sm")

    # Load new data.
    minutes = UnseenMinutes("../data/new_data", nlp)

    # Get string map of labels.
    labels = dataset.train.label_vocab._string_map

    # Initialize tokenizer.
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.backbone)

    # Intialize dictionary to store overall sentiment
    document_sentiments = {}

    # Loop over documents and classify their sentences
    for doc_name, sentences in minutes.documents.items():
        sentences_dataset = minutes.Dataset(sentences, tokenizer)
        sentences_loader = torch.utils.data.DataLoader(
            sentences_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=sentences_dataset.tokenize
        )

        all_sentence_predictions = []
        for batch in sentences_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                predictions = model(input_ids, attention_mask)
                sentence_predictions = predictions[:, labels["hawkish"]] - predictions[:, labels["dovish"]]
                all_sentence_predictions.extend(sentence_predictions.cpu().numpy())

        # Compute overall sentiment for the document by averaging sentence predictions
        overall_sentiment = np.mean(all_sentence_predictions)

        # Store overall sentiment for each document
        document_sentiments[doc_name] = overall_sentiment
    
    # Save the predicition
    pred_directory = "../predictions"
    os.makedirs(pred_directory, exist_ok=True)
    with open(os.path.join(pred_directory, "sentiment_predictions.tsv"), "w", encoding="utf-8") as file:
        for doc_name, sentiment in document_sentiments.items():
            file.write(f"{doc_name}\t{sentiment}\n")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
