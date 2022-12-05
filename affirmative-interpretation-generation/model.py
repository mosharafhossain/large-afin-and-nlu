# -*- coding: utf-8 -*-
"""
@author: Md Mosharaf Hossain
"""

import torch.nn as nn
from transformers import (
    T5ForConditionalGeneration
)

class AFINGenerator(nn.Module):
    def __init__(self, params, device, tokenizer):
        super(AFINGenerator, self).__init__()
        
        self.device = device
        self.params = params
        self.tokenizer = tokenizer
        self.transf_model = T5ForConditionalGeneration.from_pretrained(params.T5_path[params.T5_type])
        
    def forward(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self.transf_model(
            input_ids=batch["source_ids"].to(self.device),
            attention_mask=batch["source_mask"].to(self.device),
            labels=lm_labels.to(self.device),
            decoder_attention_mask=batch['target_mask'].to(self.device)
        )
        loss = outputs[0]

        return loss
    
    def predict(self, batch):
        self.eval()
        input_ids = batch["source_ids"].to(self.device)
        attention_mask = batch["source_mask"].to(self.device)
        
        
        bad_words_ids = [self.tokenizer(bad_word).input_ids for bad_word in self.params.bad_words]

        outputs = self.transf_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=self.params.num_beams,
            max_length=self.params.target_len,            
            early_stopping=self.params.early_stopping,
            do_sample=self.params.do_sample,
            top_k=self.params.top_k,
            top_p=self.params.top_p,
            repetition_penalty=self.params.repetition_penalty,
            num_return_sequences=self.params.num_return_sequences,
            bad_words_ids=bad_words_ids
        )
        
        outputs = [self.tokenizer.decode(ids) for ids in outputs]
        self.train()
        return outputs
    
