# ECG Guardian

**Team:** Dan Ben-Ezra, Lauren Albandak, Melea Ohayon

## Our question

**In PTB-XL, does Jev, with no ECG training, detect common ECG abnormalities in adults better than a model trained on ECG data ?**

AI models need millions of ECGs to learn. We want to know if a model that never learned from ECGs can do almost as well.

## Our data

We use PTB-XL. It is a free dataset with about 22,000 ECGs from 19,000 patients. Cardiologists read every ECG, so we know the right answer for each one.

The data is already split into 10 groups, and each patient is in only one group. We use groups 1 to 8 to build our method, group 9 to check it, and group 10 only once at the end for the final test.

We look at 17 common problems, like atrial fibrillation, bundle branch block and old heart attacks. PTB-XL has 71 codes, but many of them have only a few cases, so we kept the common ones.

## Two papers we read

| | Paper 1: PULSE (Liu, 2026) | Paper 2: ECG-GPT (Khunte, 2026) |
|---|---|---|
| Question | Can an AI learn to read ECG pictures? | Can an AI write the ECG diagnosis from a picture? |
| Data | More than 1 million ECG pictures | 2.9 million ECG pictures from Yale |
| Method | They trained an AI model LLaVA on ECG pictures | They trained an AI model that writes the diagnosis |
| Validation | Tested on other datasets | Tested in 7 other hospitals |
| Result | Works well after training. AI models without training were almost random | Very good. AF score 0.956 on PTB-XL |
| Left open | Needs a lot of data to train | Needs millions of ECGs. We can't use the model |

**What is missing:** both papers need millions of ECGs to train. Nobody tested a model with no ECG training that gets measurements instead of a picture.

## What we do

1. Our code measures the ECG (heart rate, is the rhythm regular, QRS width…).
2. We write the measurements as words, like "irregular rhythm".
3. Jev reads the words and gives a number from 0 to 1 for each problem (how likely it is). Jev never sees the right answer.
4. We compare Jev with:
   - **XGBoost**, a model we will train on groups 1–8 with the same measurements. It will be our main comparison, because it will be tested on the same ECGs as Jev.
   - **ECG-GPT**, a model trained on 2.9 million ECGs. We use its published results on PTB-XL, only as a reference, because we can't run it on our ECGs.
   - **Simple rules**, textbook criteria written in code, with no AI and no training. For example: heart rate under 60 = slow heart rate; very irregular rhythm = atrial fibrillation; wide QRS = bundle branch block.
5. We use AUROC to give a score.

## First results (group 9, 2,183 ECGs)

Jev got between 0.75 and 0.99 on the 17 problems.

| Problem | Jev | Simple rule |
|---|---|---|
| Atrial fibrillation | 0.975 | 0.937 |
| Right bundle branch block | 0.986 | 0.973 |
| Left bundle branch block | 0.933 | 0.912 |
| Slow heart rate | 0.895 | 0.963 |
| 1st-degree AV block | 0.750 | 0.750 |

For atrial fibrillation, Jev did better than a simple rule and close to ECG-GPT (0.956). It is weaker on some problems, like slow heart rate. We still need to train XGBoost to compare.

## Plan

| Date | What we do |
|---|---|
| 2 Dec | Requirements file |
| 9 Dec | Design file and EDA |
| 16 Dec | Task list and train XGBoost |
| 23 Dec | Tests and compare Jev with XGBoost |
| 30 Dec | Final test on group 10 and final report |
| 6 / 13 Jan | Presentation |

## Use of AI

We used Claude Code to help with the code and to understand the papers. Jev is the model we test.
