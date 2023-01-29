import torch.nn as nn


DEFAULT_DROPOUT = 0.2
DEFAULT_HIDDEN_SIZE = 512


class BERT_Fake(nn.Module):

    def __init__(self, bert, dropout=DEFAULT_DROPOUT, hidden_size=DEFAULT_HIDDEN_SIZE) -> None:
        super(BERT_Fake, self).__init__()

        self.bert = bert
        # DropoutLayer
        self.dropout = nn.Dropout(dropout)
        # ReLU
        self.relu = nn.ReLU()
        # Dense Layers
        self.fc1 = nn.Linear(768, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size)
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
