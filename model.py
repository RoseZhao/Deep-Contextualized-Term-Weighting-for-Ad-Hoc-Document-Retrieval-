import torch.nn as nn
from transformers import BertModel, XLNetModel


class HDCTModel(BertModel):
    """
    A model wrapper around hf BERT implementation
    """
    def __init__(self, config):
        super().__init__(config=config)
        self.fc = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = super().forward(input_ids, attention_mask, token_type_ids)
        last_hidden = outputs[0]    # (batch_size, seq_len, hidden_size)
        logits = self.fc(last_hidden)    # (batch_size, seq_len, 1)
        return logits


class XLNETModel(XLNetModel):
    """
    A model wrapper around hf XLNET implementation
    """
    def __init__(self, config):
        super().__init__(config=config)
        self.fc = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = super().forward(input_ids=input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        last_hidden = outputs[0]    # (batch_size, seq_len, hidden_size)
        logits = self.fc(last_hidden)    # (batch_size, seq_len, 1)
        return logits

