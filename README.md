# Clinical Trials Eligibility Prediction

This repository contains code and resources for predicting clinical trial eligibility based on patient summaries and trial inclusion/exclusion criteria.  
It includes preprocessing scripts, model architectures, and evaluation utilities, along with a quick demo script.

## Installation

Clone this repository and install the dependencies:

```bash
git clone https://github.com/imnishitha/Clinical-Trials-Eligibility
cd Clinical-Trials-Eligibility
pip install -r requirements.txt

Recommended Python version: 3.10.17

Demo
A simple demo is provided to showcase the model's functionality.
To run the demo:

python demo.py

The demo uses a small sample dataset (demo_data.json) already included in the repository.
You can add more examples to this file in the same format for custom testing.

Repository Structure
Clinical-Trials-Eligibility/
├── Analysis Scripts
├── BPE
├── Dataset/               # Dataset files (demo_data.json, etc.)
├── Helper_scripts         # Additional Utilities
├── PyTorch_Files/         # Model classes
├── results/               # Saved plots and metrics
├── demo.py                # Demo script
├── requirements.txt       # Dependencies
└── README.md

Pre-Requisites
Before running the code, ensure you have:
Python 3.10.17 installed.
pip package manager installed.
All dependencies from requirements.txt installed.