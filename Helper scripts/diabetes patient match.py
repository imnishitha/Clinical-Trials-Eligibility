import json

with open("PMC-Patients_human_eval.json", "r") as file:
    patients = json.load(file)

# Inclusion criteria keywords
inclusion_keywords = [
    "type 1 diabetes", "type 2 diabetes", "diabetes mellitus",
    "18 years", "19 years", "20 years", "adult", "older than 18", "age of 18",
    "admitted", "hospitalized", "surgical service", "medical service",
    "oral hypoglycemic", "insulin", "diet alone", "insulin pump", "sq insulin",
    "continuous glucose monitoring", "CGM", "sensor placed", "monitor on admission"
]

# Exclusion criteria keywords
exclusion_keywords = [
    "under 18", "child", "pediatric", "17 years", "16 years", "15 years",
    "COVID-19", "coronavirus", "positive for covid", "SARS-CoV-2",
    "infection of the skin at the cgm site", "skin infection", "sensor removal",
    "altered mental status", "confusion", "disoriented",
    "unable to scan", "cannot scan", "inability to scan", "less than 24 hours", 
    "short stay", "overnight only", "unable to provide consent", "no consent"
]




# inclusion_keywords = [
#     "diabetes mellitus type 1", "diabetes mellitus type 2", "type 1 diabetes", "type 2 diabetes", "diabetic",
#     "follow-up", "followed up", "under follow-up", "follow up in same center", "followed in our center"
# ]

# exclusion_keywords = [
#     "decompensated liver disease", "hepatic failure", "advanced liver disease",
#     "psychiatric disorder", "mental illness", "cognitive impairment", "schizophrenia", "bipolar", "psychosis",
#     "bariatric surgery", "gastric bypass", "weight loss surgery",
#     "renal replacement therapy", "dialysis", "hemodialysis", "peritoneal dialysis"
# ]



# Simple helper to check presence of keywords
def contains_keywords(text, keywords):
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)

# Filter logic
eligible_patients = []
for patient in patients:
    text = patient.get("patient", "")
    
    includes = contains_keywords(text, inclusion_keywords)
    excludes = contains_keywords(text, exclusion_keywords)

    if includes and  excludes:
        eligible_patients.append({
            "id": patient["human_patient_id"],
            "uid": patient["human_patient_uid"],
            "PMID": patient.get("PMID", ""),
            "summary": text[:300] + "..."
        })

print(f"Eligible patient count: {len(eligible_patients)}\n")
# for p in eligible_patients:
#     print(f"ID: {p['id']} | PMID: {p['PMID']} | UID: {p['uid']}")
#     print(f"Summary: {p['summary']}")
#     print("-" * 80)
