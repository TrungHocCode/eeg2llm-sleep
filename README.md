# eeg2llm-sleep
A pipeline for analyzing sleep EEG data using fine-tuned language models.
## Overview
eeg2llm-sleep is a tool that processes EEG sleep recordings and analyzes them using a fine-tuned GPT-2 language model. The pipeline converts EEG data into a format that can be interpreted by the language model to provide insights into sleep patterns and anomalies.
## Features

Process EDF (European Data Format) sleep recordings
Convert EEG signals into language model-compatible input
Use features extract from EEG model to diagnose
Analyze sleep patterns using a fine-tuned GPT-2 model
Generate reports and visualizations of sleep analysis

## Prerequisites

Python 3.7+
Required Python packages (see requirements.txt)
Sufficient disk space for model storage

## Installation

1. Clone this repository:
git clone https://github.com/TrungHocCode/eeg2llm-sleep.git
cd eeg2llm-sleep

2. Install the required dependencies:
pip install -r requirements.txt

3. Download the fine-tuned GPT-2 model:

Follow this link to download the model: "https://drive.google.com/drive/folders/1Objfem_CUSrHjN9gtYN3Dre0dmIRTH-T?usp=drive_link"
Place the downloaded model folder in the results directory