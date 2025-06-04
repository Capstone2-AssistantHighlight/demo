# model.py

import torch
import torch.nn as nn
from transformers import BertModel

class KoBERTClassifier(nn.Module):
    """
    KoBERT 기반 문장 분류 모델.
    BERT base 위에 dropout + linear 레이어(클래스 개수)로 구성함.
    """
    def __init__(self, bert_model: BertModel, dr_rate=0.5, num_classes=7):
        super(KoBERTClassifier, self).__init__()
        self.bert = bert_model  # 이미 load된 KoBERT BertModel
        self.dropout = nn.Dropout(p=dr_rate)
        # BERT hidden size는 768 (KoBERT base 기준)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, token_ids, valid_length, segment_ids):
        """
        Args:
            token_ids (torch.LongTensor): [batch_size, seq_len]
            valid_length (torch.LongTensor): [batch_size]
            segment_ids (torch.LongTensor): [batch_size, seq_len]
        Returns:
            logits (torch.FloatTensor): [batch_size, num_classes]
        """
        # BERT 출력: (sequence_output, pooled_output)
        # pooled_output: [batch_size, hidden_size]중 [CLS] 토큰과 대응
        _, pooled_output = self.bert(input_ids=token_ids, token_type_ids=segment_ids)
        # (또는: out = self.bert(token_ids, token_type_ids=segment_ids)[1])
        out = self.dropout(pooled_output)
        logits = self.classifier(out)  # [batch_size, num_classes]
        return logits
