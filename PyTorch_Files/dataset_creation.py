import torch
from torch.utils.data import Dataset

class DataCreator(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.patient = list(df['patient'])
        self.criteria = list(df['criteria'])
        self.labels = list(df['label'])
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        combined_text = "[CLS]" + self.patient[index] + "[SEP]" + self.criteria[index]
        encoding = self.tokenizer.encode(combined_text)

        # Convert the Encoding to Tensors
        ids = encoding.ids[:self.max_len]
        ids += [self.tokenizer.token_to_id("[PAD]")] * (self.max_len - len(ids))

        attention_mask = [1 if i!=self.tokenizer.token_to_id("[PAD]") else 0 for i in ids]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(self.labels[index], dtype=torch.long)
        }
