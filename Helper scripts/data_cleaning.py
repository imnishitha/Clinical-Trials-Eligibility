import string
import pandas as pd


"""
This file takes the consolidated sentences, replaces hyphens, and removes rest of the punctuation.
This step is done in order to ensure clear vocabulary creation during BPE training.
"""

def clean_text(text):
    text = text.lower()
    text = text.replace("-", " ")
    text = "".join([word for word in text if word not in string.punctuation])
    text = text.strip()
    # text = text.split()
    # text = word_tokenize(text)
    return text

df = pd.read_csv('./Dataset/consolidated_data_2.csv')
df['patient'] = df['patient'].apply(lambda x: clean_text(x))
df['criteria'] = df['criteria'].apply(lambda x: clean_text(x))
df.to_csv('./Dataset/cleaned_data_3.csv', header=True, index=False)
print("Data Cleaning Succesful; New Data File exported to Dataset folder!") 