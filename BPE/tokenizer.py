import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

df = pd.read_csv('./Dataset/cleaned_data_3.csv')

corpus = df['patient'].astype(str) + " " + df['criteria'].astype(str)
corpus = corpus.tolist()

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()

vocab_size = 30000
trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

print(f"BPE tokenizer training started with vocab size: {vocab_size}...")
tokenizer.train_from_iterator(corpus, trainer=trainer)
tokenizer.save("bpe_tokenizer.json")

print(f"Trained tokenizer saved to 'bpe_tokenizer.json' with a vocab size of {vocab_size}.")
