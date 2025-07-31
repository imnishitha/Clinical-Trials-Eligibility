import pandas as pd
import torch
import toml
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from sklearn.model_selection import train_test_split

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

df = pd.read_csv('./Dataset/cleaned_data_3.csv')
tokenizer = Tokenizer.from_file('./BPE/bpe_tokenizer.json')

config = toml.load('config.toml')

dataset = DataCreator(df=df, tokenizer=tokenizer, max_len=8) 
print(df['patient'][0][:50])
print(dataset[0])

train_df, temp_df = train_test_split(df, test_size=0.7, random_state=config["general"]["seed"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=config["general"]["seed"])

train_data = DataCreator(df=train_df, tokenizer=tokenizer, max_len=config["model"]["max_len"])
val_data = DataCreator(df=val_df, tokenizer=tokenizer, max_len=config["model"]["max_len"])
test_data = DataCreator(df=test_df, tokenizer=tokenizer, max_len=config["model"]["max_len"])

train_loader = DataLoader(dataset=train_data, batch_size=config["training"]["batch_size"], shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=config["training"]["batch_size"], shuffle=False)
test_loader = DataLoader(dataset=test_data, batch_size=config["training"]["batch_size"], shuffle=False)

print("Dataset and DataLoaders created successfully!")