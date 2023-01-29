import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, RandomSampler, DataLoader

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn import metrics
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split

from skopt import gp_minimize
from sklearn.metrics import roc_auc_score


import transformers
from transformers import AutoTokenizer, AutoModel

from helpers import load_dataset
from helpers import plot_roc_curve
from helpers import generate_metrics
from helpers import plot_confusion_matrix
from helpers import preprocess_titles
from helpers import strip_accents_and_lowercase

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:512'

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

DATASET = 'greek'
BATCH_SIZE = 64
EPOCHS = 15
DROPOUT = 0.2
HIDDEN_SIZE = 512
COMMENT = 'LR4'
RUN_ID = f'{DATASET}_{BATCH_SIZE}_{EPOCHS}_{DROPOUT}_{HIDDEN_SIZE}_{COMMENT}'
WEIGHTS_SAVE_PATH = f'weights/w_{RUN_ID}.pt'
OUTPUTS_PATH = f'outputs/Neural_Networks/Mine/{RUN_ID}'

# create output directory for current run
Path(OUTPUTS_PATH).mkdir(exist_ok=True)

# suppress noisy warnings
transformers.logging.set_verbosity_error()

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

# pd.Series(num_of_words_per_title).hist(bins=30)
# max_seq_len = 25

# tokenization and encoding of the sequences in the training and testing set
tokens_train = tokenizer.batch_encode_plus(train_title.tolist(),
                                           max_length=max_seq_len,
                                           padding='max_length',
                                           truncation=True,
                                           return_token_type_ids=False)

tokens_test = tokenizer.batch_encode_plus(test_title.tolist(),
                                          max_length=max_seq_len,
                                          padding='max_length',
                                          truncation=True,
                                          return_token_type_ids=False)

# Convert Integer Sequences to Tensors
# For train set
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

# For test set
test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())


# wrap tensors
# the TensorDataset is a ready to use class to represent our data as a list of tensors
train_data = TensorDataset(train_seq, train_mask, train_y)

# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

train_dataloader = DataLoader(train_data,
                              sampler=train_sampler,
                              batch_size=BATCH_SIZE)

# Freeze BERT parameters
for param in lm_model_greek.parameters():
    param.requires_grad = False


# Define the Model Custom Architecture
class BERT_Fake(nn.Module):

    def __init__(self, bert) -> None:
        super(BERT_Fake, self).__init__()

        self.bert = bert
        # DropoutLayer
        self.dropout = nn.Dropout(DROPOUT)
        # ReLU
        self.relu = nn.ReLU()
        # Dense Layers
        self.fc1 = nn.Linear(768, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, 1)
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(HIDDEN_SIZE)
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

# Pass the pre-trained BERT to our custom architecture
model = BERT_Fake(lm_model_greek)

# Push model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

criterion = nn.BCELoss()
criterion = criterion.to(device)

writer = SummaryWriter('runs/greek-fake-news')

def train():
    model.train()
    total_loss = 0
    total_preds = []

    # Iterate over the batches
    for step, batch in enumerate(train_dataloader):
        # Update after every Number of batches
        if step % 100 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step,
                                                       len(train_dataloader)))

        batch = [r.to(device) for r in batch]
        send_id, mask, labels = batch
        labels = labels.type(torch.LongTensor) 
        labels = labels.to(device)

        # Forward
        outputs = model(send_id, mask)
        labels = labels.view(-1, 1)
        loss = criterion(outputs, labels.float())

        # Backward
        # Make the grads zero
        model.zero_grad()
        # Do the backward step of the loss calculation through chain derivatives
        total_loss += loss.item()
        loss.backward()

        # Clip the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        # Do the optimizer step and update the parameters
        optimizer.step()

        # Model predictions are stored on GPU. Push it to CPU
        outputs = outputs.detach().cpu().numpy()

        # Append the model predictions
        total_preds.append(outputs)

    # Compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # The predictions are in the form of (no. of batches, size of batch, no. of classes)
    # Reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    # Return the loss and predictions
    return avg_loss, total_preds


# model training
for epoch in range(EPOCHS):
    print('Epoch {:} / {:}'.format(epoch + 1, EPOCHS))
    train_loss, preds = train()

    # To find out what happens with the accuracy per epoch
    writer.add_scalar('Training Loss', train_loss, epoch)

    print(f'Training Loss: {train_loss:.3f}')
torch.save(model.state_dict(), WEIGHTS_SAVE_PATH)

writer.close()

# model.load_state_dict(torch.load(WEIGHTS_SAVE_PATH))

# Get Predictions for the Test Data
with torch.no_grad():
    preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu()

# define the objective function
def objective(threshold):
    predictions = (preds > threshold[0]).float()
    auc = roc_auc_score(test_y, predictions)
    return -auc

# optimize the threshold
res = gp_minimize(objective, [(0, 1.0)], n_calls=30)
best_threshold = res.x[0]

# threshold the output
preds = (preds > best_threshold).float()
conf_matrix = metrics.confusion_matrix(test_y, preds)

roc_name = f'{OUTPUTS_PATH}/roc_'
metrics_name = f'{OUTPUTS_PATH}/metrics.txt'
cm_name = f'{OUTPUTS_PATH}/cm.jpg'

generate_metrics(conf_matrix, save_path=metrics_name, save=True, verbose=True)
plot_confusion_matrix(conf_matrix,
                      classes=['Real Data', 'Fake Data'],
                      save_path=cm_name,
                      save=True
)
plot_roc_curve(test_y, preds, roc_name=roc_name, save=True)
