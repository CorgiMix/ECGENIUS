# ECG Guardian

**Team:** Dan Ben-Ezra, Lauren Albandak, Melea Ohayon

## Our question

**In PTB-XL, how does Jev, with no ECG training, compare with trained models for detecting 17 common ECG abnormalities, and where is each one best?**

AI models usually need many ECGs to learn. We want to compare models that learned from very different amounts of data, from none (Jev) to millions of ECGs (ECGFounder), and see where each one works best.

## Our data

We use PTB-XL. It is a free dataset with about 22,000 ECGs from 19,000 patients. Cardiologists read every ECG, so we know the right answer for each one. We already downloaded the data.

The data is already split into 10 groups, and each patient is in only one group. We use groups 1 to 8 (17,418 ECGs) to train the models, group 9 to check them, and group 10 only once at the end for the final test.

We look at 17 common problems in 3 types:
- **Rhythm:** atrial fibrillation, extra beats (PVC, PAC), slow heart rate, fast heart rate, sinus arrhythmia
- **Conduction:** left and right bundle branch block, incomplete right bundle branch block, 1st-degree AV block, left anterior fascicular block
- **Structure:** old heart attacks (3 locations), enlarged left ventricle, ischemia, ST depression

PTB-XL has 71 codes, but many of them have only a few cases, so we kept the common ones.

## Two papers we read

| | Paper 1: PULSE (Liu, 2026) | Paper 2: ECG-GPT (Khunte, 2026) |
|---|---|---|
| Question | Can an AI learn to read ECG pictures? | Can an AI write the ECG diagnosis from a picture? |
| Data | More than 1 million ECG pictures | 2.9 million ECG pictures from Yale |
| Method | They trained an AI model (LLaVA) on ECG pictures | They trained an AI model that writes the diagnosis |
| Validation | Tested on other datasets | Tested in 7 other hospitals |
| Result | Works well after training. AI models without training were almost random | Very good. AF score 0.956 on PTB-XL |
| Left open | Needs a lot of data to train | Needs millions of ECGs. We can't use the model |

Links: https://doi.org/10.1038/s41746-026-02551-3 · https://doi.org/10.1093/ehjdh/ztag031

**What is missing:** both papers need millions of ECGs to train. Nobody compared, on the same ECGs, a model with no ECG training against models trained on a little or a lot of data, problem by problem.

## The 4 models we compare

| Model | Trained on ECGs? | What it gets |
|---|---|---|
| **Jev** (our model) | No | Our measurements written as words |
| **Gradient boosting (XGBoost)** | Yes, on groups 1–8 | Our measurements as numbers |
| **CNN from scratch** | Yes, on groups 1–8 only | The raw ECG signal |
| **ECGFounder + new head** | Yes, on 10.7 million ECGs (Harvard-Emory), then we train a new last layer on groups 1–8 | The raw ECG signal |

ECGFounder was not trained on PTB-XL, so it has not seen our test ECGs.

## What we do

1. **Pre-processing:** we prepare the ECG signals for the CNN and ECGFounder (500 Hz, 10 seconds, 12 leads, filtering, normalization), in the format each model needs.
2. **Jev:** our code measures the ECG (heart rate, is the rhythm regular, QRS width…) and writes the measurements as words, like "irregular rhythm". Jev reads the words and gives a number from 0 to 1 for each problem. Jev never sees the right answer.
3. **Trained models:** we train XGBoost, the CNN and the new head of ECGFounder on groups 1–8.
4. **Compare:** we use AUROC for each of the 17 problems and for each type (rhythm, conduction, structure), to see where each model is best.
5. **Explain:** for each model, we explain its architecture: how it works and why.

We use ECG-GPT's published results only as a reference, because we can't run it on our ECGs.

## First results (group 9, 2,183 ECGs)

We already ran Jev. It got between 0.75 and 0.99 on the 17 problems. For example: atrial fibrillation 0.975, right bundle branch block 0.986, 1st-degree AV block 0.750. The 3 trained models are the next step.

## Plan

| Date | What we do |
|---|---|
| 2 Dec | Requirements file; set up ECGFounder |
| 9 Dec | Design file; EDA; pre-processing |
| 16 Dec | Task list; train XGBoost and the CNN |
| 23 Dec | Tests; train the ECGFounder head; compare the 4 models |
| 30 Dec | Final test on group 10; final report |
| 6 / 13 Jan | Presentation, with the architecture of each model |

## Use of AI

We used Claude Code to help with the code and to understand the papers. Jev is one of the models we test.
