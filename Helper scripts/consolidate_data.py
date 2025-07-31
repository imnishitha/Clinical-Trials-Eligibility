import pandas as pd
import numpy as np
import random

df = pd.read_csv('./Dataset/original_json_data_1.csv')
syn = pd.read_csv('./Dataset/synthetic_data_binary_0.csv')
syn.drop(columns=['index'], inplace=True)

subset_one = df[df['label']=='negative']
excluded_indices = subset_one.index
all_indices = list(range(0, df.shape[0]))

# Sampling 200 (can change) neutral examples to balance our dataset
available_indices = [i for i in all_indices if i not in excluded_indices]
sampled_indices = random.sample(available_indices, 200)

# Concatenating with the Synthetic Data 
subset = pd.concat([syn, subset_one, df.loc[sampled_indices, :]])
subset.reset_index(drop=True, inplace=True)
subset['label'] = subset['label'].astype(str)

# Convert the labels to floats for Model Training ease
"""
True -> Positive -> 2
False -> Negative -> 0
Neutral/Unknown -> Neutral -> 1
"""
def convert_labels(label):
    if label == 'True':
        ans = int(2)
    elif label == 'unknown':
        ans = int(1)
    else:
        ans = int(0)
    return ans

subset['label'] = subset['label'].apply(lambda x: convert_labels(x))
subset.to_csv('./Dataset/consolidated_data_2.csv', index=False, header=True)

print("Consolidated Data Generated Successfully")