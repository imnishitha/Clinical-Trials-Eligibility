import os
import json
import pandas as pd
import warnings
warnings.simplefilter("ignore")

data_path = './Synthetic_Data/'

# Data-Frame Creation from Individual JSON Files
df = pd.DataFrame(columns=['patient', 'criteria', 'label'])

with open('./Dataset/criteria.json', 'r') as file:
    criteria = json.load(file)

for filename in os.listdir(data_path):
    if filename.endswith('.json'):
        file_path = os.path.join(data_path, filename)
        print(f"Processing {file_path}")

        with open(file_path, 'r') as file:
            data = pd.read_json(file)
        sub_data = data[['patient']]
        
        # Get the JSON_filename for the respective criteria
        json_key = filename.split('.')[0]
        crit = [criteria[json_key]]*sub_data.shape[0]
        sub_data['criteria'] = crit
        sub_data['label'] = data['match_criteria']
        df = pd.concat([df, sub_data])
        
        print(f"File {json_key} Processed")

df.reset_index(inplace=True, drop=False)
df.to_csv('./Dataset/synthetic_data_binary.csv', header=True, index=False)
print("Synthetic Data Processed and Generated Dataframe succesfully")