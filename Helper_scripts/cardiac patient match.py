import json

with open("PMC-Patients_human_eval.json", "r") as file:
    patients = json.load(file)

inclusion_keywords = [
    "out-of-hospital cardiac arrest", "return of spontaneous circulation", "ROSC",
    "ventricular fibrillation", "ventricular tachycardia", "VF", "VT",
    "Glasgow Coma Scale", "GCS ≤ 8", "GCS score of 8", "GCS of 8", "GCS score 8"
]

exclusion_keywords = [
    "terminal renal insufficiency", "end-stage renal disease",
    "G6PD deficiency", "glucose 6-phosphate dehydrogenase deficiency",
    "urolithiasis", "oxalate nephropathy", "hemochromatosis",
    "treatment limitation", "do not resuscitate", "DNR"
]


def contains_keywords(text, keywords):
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)

eligible_patients = []
for patient in patients:
    text = patient.get("patient", "")

    includes = contains_keywords(text, inclusion_keywords)
    excludes = contains_keywords(text, exclusion_keywords)

    if includes and not excludes:
        eligible_patients.append({
            "id": patient["human_patient_id"],
            "uid": patient["human_patient_uid"],
            "PMID": patient.get("PMID", ""),
            "summary": text[:300] + "..."
        })

print(f"Eligible patient count: {len(eligible_patients)}\n")
