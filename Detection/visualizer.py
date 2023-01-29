import torch
from torch import nn
import torch.onnx
from torchviz import make_dot
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import TensorDataset, RandomSampler, DataLoader

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import transformers
from transformers import AutoTokenizer, AutoModel

from helpers import load_dataset
from helpers import strip_accents_and_lowercase


import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'

lm_model_greek = AutoModel.from_pretrained(
    'nlpaueb/bert-base-greek-uncased-v1', return_dict=False)
tokenizer = AutoTokenizer.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')


class BERT_Fake(nn.Module):

    def __init__(self, bert) -> None:
        super(BERT_Fake, self).__init__()

        self.bert = bert
        # DropoutLayer
        self.dropout = nn.Dropout(0.2)
        # ReLU
        self.relu = nn.ReLU()
        # Dense Layers
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 1)
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(512)
        self.sigmoid = nn.Sigmoid()

    def forward(self, send_id, mask):
        _, cls_hs = self.bert(send_id, attention_mask=mask)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Define your model
model = BERT_Fake(lm_model_greek)
DATASET = 'greek'

# -----------------
# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')
lm_model_greek = AutoModel.from_pretrained(
    'nlpaueb/bert-base-greek-uncased-v1', return_dict=False)

# Load dataset and preprocess
dataset_df: pd.DataFrame = load_dataset(DATASET)
# titles = dataset_df['title'].apply(preprocess_titles)
titles = dataset_df['title'].apply(strip_accents_and_lowercase)
labels = dataset_df['is_fake']

# split dataset to test and train data
train_title, test_title, train_labels, test_labels = train_test_split(
    titles, labels, random_state=2023, test_size=0.3, stratify=labels)

# ------ Tokenization ------
# Get the number of words for every title in the train set
num_of_words_per_title = [len(i.split()) for i in train_title]
max_seq_len = int(np.percentile(num_of_words_per_title, 75))

# tokenization and encoding of the sequences in the training and testing set
tokens_train = tokenizer.batch_encode_plus(train_title.tolist(),
                                           max_length=max_seq_len,
                                           padding='max_length',
                                           truncation=True,
                                           return_token_type_ids=False)


# Create a dummy input for the model
# tokenization and encoding of the sequences in the training and testing set
tokens_train = tokenizer.batch_encode_plus(train_title.tolist(),
                                           max_length=max_seq_len,
                                           padding='max_length',
                                           truncation=True,
                                           return_token_type_ids=False)

# suppress noisy warnings
transformers.logging.set_verbosity_error()

train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

train_data = TensorDataset(train_seq, train_mask, train_y)

# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

train_dataloader = DataLoader(train_data,
                              sampler=train_sampler,
                              batch_size=32)

dot = make_dot(model(train_seq, train_mask), params=dict(model.named_parameters()))

# Save the dot object as an image file
dot.render("outputs/BertFake.pdf")
