import json
import pandas as pd

# Load patients
with open("./Dataset/Neutral_data.json", "r") as file:
    patients = json.load(file)

# Constant criteria JSON
with open("./Dataset/criteria.json", "r") as file:
    criteria_json = json.load(file)

def extract_keywords(criteria_text):
    parts = criteria_text.split("Exclusion Criteria:")
    inclusion_text = parts[0].replace("Inclusion Criteria:", "").strip()
    exclusion_text = parts[1].strip()
    return [kw.strip() for kw in inclusion_text.split(",")], [kw.strip() for kw in exclusion_text.split(",")]

def contains_keywords(text, keywords):
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)

# Add new key "match_criteria" for each patient
for patient in patients:
    patient_text = patient.get("patient", "")
    match_results = []

    for trial_id, crit_text in criteria_json.items():
        inc_keywords, exc_keywords = extract_keywords(crit_text)

        text_lower = patient_text.lower()

        has_excluded = any(keyword.lower() in text_lower for keyword in exc_keywords)
        has_included = any(keyword.lower() in text_lower for keyword in inc_keywords)

        if has_excluded:
            label = "negative"
        elif has_included:
            label = "positive"
        else:
            label = "unknown"

        match_results.append({"criteria": trial_id, "label": label})

    patient["match_criteria"] = match_results

print("Results and Criteria added to Each Patient")

# Consolidating results into a dataframe
df = pd.DataFrame(columns=['patient', 'criteria', 'label'])

for patient in patients:
    patient_text = patient.get("patient", "")
    for mc in patient['match_criteria']:
        crit = criteria_json[mc['criteria']]
        label = mc['label']
        df.loc[df.shape[0], :] = [patient_text, crit, label]

df.to_csv('./Dataset/original_json_data_1.csv', index=False, header=True)
print("Patient DataFrame Created Successfully")