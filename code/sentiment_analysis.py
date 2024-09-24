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
parser.add_argument("--batch_size", default=16, type=int, help="Batch size used for training.")
parser.add_argument("--epochs", default=5, type=int, help="Number of training epochs.")
parser.add_argument("--seed", default=17, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--backbone", default="bert-large-uncased", type=str, help="Pre-trained transformer.")
parser.add_argument("--learning_rate", default=1e-05, type=float, help="Learning rate.")
parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay.")

# FIXME: Add more arguments if needed (e.g., dropout rate, size of dense layer, etc.).

# Create the Model class.
class Model(TrainableModule):
    def __init__(self, args, backbone, dataset):
        super().__init__()
        # Initialize Model class by defining it.
        self._backbone = backbone
        self._dense = torch.nn.Linear(backbone.config.hidden_size, backbone.config.hidden_size * 2)
        self._dropout = torch.nn.Dropout(args.dropout)
        self._activation = torch.nn.ReLU()
        self._classifier = torch.nn.Linear(backbone.config.hidden_size * 2, len(dataset.label_vocab))

    # Implement the model computation.
    def forward(self, input_ids, attention_mask):
        hidden = self._backbone(input_ids, attention_mask).last_hidden_state
        hidden = self._dense(hidden)
        hidden = self._activation(hidden)
        hidden = self._dropout(hidden)
        hidden = self._classifier(hidden)
        hidden = torch.mean(hidden, dim=1)
        return hidden

def main(args):
    # Set the random seed and number of threads.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Creat logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the transformer model and its tokenizer
    backbone = transformers.AutoModel.from_pretrained(args.backbone)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.backbone)

    # Load the data
    minutes = CBMinutesDataset("../data")

    # Create the dataloaders
    def prepare_example(example):
        return example["document"], example["label"]
    
    def prepare_batch(data):
        documents, labels = zip(*data)
        tokenized_docs = tokenizer(documents, padding="longest", return_attention_mask=True)
        input_ids, attention_mask = tokenized_docs["input_ids"], tokenized_docs["attention_mask"]
        label_ids = minutes.train.label_vocab.indices(labels)
        return (torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)), torch.tensor(label_ids, dtype=torch.long)
    
    def create_dataloader(dataset, shuffle):
        return torch.utils.data.DataLoader(
            dataset.transform(prepare_example), args.batch_size, shuffle, collate_fn=prepare_batch
        )
    
    # Create dataloader objects from datasets
    train = create_dataloader(minutes.train, shuffle=True)
    dev = create_dataloader(minutes.dev, shuffle=False)
    test = create_dataloader(minutes.test, shuffle=False)
    
    # Create the model
    model = Model(args, backbone, minutes.train)
    optimizer = torch.optim.AdamW(backbone.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train) * args.epochs
    schedule = transformers.get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps,
    )

    # Configure model and train
    model.configure(
        optimizer=optimizer,
        schedule=schedule,
        loss=torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1),
        metrics=torchmetrics.Accuracy(
            task="multiclass", num_classes=len(minutes.train.label_vocab), ignore_index=tokenizer.pad_token_id
            ),
        logdir=args.logdir,
    )
    
    # def predict_callback(self, epoch, logs):
    #     model.predict(dev)
    #     # Generate test set annotations, but in 'args.logdir' to allow for parallel execution
    #     os.makedirs(args.logdir, exist_ok=True)
    #     with open(os.path.join(args.logdir, f"sentiment_analysis{epoch}.txt"), "w", encoding="utf-8") as prediction_file:
    #         # Predict the tags on the test set
    #         predictions = model.predict(test)
    #         for sentence in predictions:
    #             print(minutes.train.label_vocab.string(int(np.argmax(sentence))), file=prediction_file)

    # Fit the model to the data
    model.fit(train, dev=dev, epochs=args.epochs)#, callbacks=[predict_callback])

    # Generate test set annotations, but in 'args.logdir' to allow for parallel execution
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "sentiment_analysis.txt"), "w", encoding="utf-8") as predictions_file:
        # Predict the tags on the test set.
        predictions = model.predict(test)
        for sentence in predictions:
            print(minutes.train.label_vocab.string(int(np.argmax(sentence))), file=predictions_file)
    
if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)