import torch

import transformers
from transformers import AutoTokenizer, AutoModel

from bert_fake_model import BERT_Fake


WEIGHTS_PATH = 'weights/bert_fake.pt'

# suppress noisy warnings
transformers.logging.set_verbosity_error()

tokenizer = AutoTokenizer.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')
lm_model_greek = AutoModel.from_pretrained(
    'nlpaueb/bert-base-greek-uncased-v1', return_dict=False)
    
model = BERT_Fake(lm_model_greek)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH))
model.eval()


def calculate_possibility(input_text: str) -> float:
    tokens_test = tokenizer.batch_encode_plus([input_text],
                                        max_length=40,
                                        padding='max_length',
                                        truncation=True,
                                        return_token_type_ids=False)

    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])

    with torch.no_grad():
        pred = model(test_seq.to(device), test_mask.to(device))
        pred = pred.detach().cpu().numpy()
    return pred[0][0]