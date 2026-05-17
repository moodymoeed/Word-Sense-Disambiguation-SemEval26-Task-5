# CS445 Group 2: Word Sense Disambiguation (SemEval-26 Task 5)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-F9D371.svg)](https://huggingface.co/)

**Official repository for CS445 Group 2. We tackle SemEval-2026 Task 5 (AmbiStory) by predicting continuous word sense plausibility using a Flat DeBERTa-v3-base architecture with Multi-Objective Loss and Nelder-Mead Threshold Optimization.**

---

## Project Overview

Traditional Word Sense Disambiguation (WSD) treats language as a multiple-choice test. However, in real-world contexts, multiple interpretations of a word can remain plausible simultaneously. **SemEval-2026 Task 5** models this phenomenon using the **AmbiStory dataset**, requiring systems to predict human plausibility ratings (on a continuous scale from 1 to 5) for candidate word senses given a 5-sentence narrative.

Instead of relying on closed-source LLM APIs, our group developed a self-contained deep learning methodology. We underwent extensive experimentation, shifting from complex hierarchical models to a highly optimized flat-transformer approach that mathematically balances relative ranking with absolute accuracy.

## Architecture Plan and Implementation Evolution

**Initial Plan:** Based on our initial literature review, we planned to implement a **Hierarchical LoRA-adapted DeBERTa + BiLSTM** architecture. We hypothesized that encoding the sentence, precontext, and ending independently before feeding them into a sequence model would best capture the chronological flow of the narrative.

**Final Implementation:** We ultimately shifted to a **Flat DeBERTa-v3-base** architecture. 

**Why we shifted:**
1. **Unnecessary Complexity:** The AmbiStory dataset texts are extremely short (5 sentences maximum). The hierarchical structure added computational overhead and caused severe overfitting without improving the model's ability to cross-attend across sentences.
2. **Self-Attention Superiority:** Flat concatenation allowed the self-attention mechanism within the DeBERTa layers to build global context between the sentence, the ending, and the judged meaning seamlessly from the ground up.
3. **Training Stability:** The hierarchical model took significantly longer to train and was highly unstable. The flat transformer proved to be far more robust, memory-efficient, and achieved much higher evaluation metrics.

## Final System Architecture

Our final implemented pipeline consists of four main stages:

1. **Flat Context Encoding (DeBERTa-v3-base):** 
   We concatenate the `precontext`, `sentence`, `ending`, and `judged_meaning` separated by special `[SEP]` tokens. We use `microsoft/deberta-v3-base` to encode the entire sequence, fully fine-tuning the base model parameters.
   
2. **Dense Regression Head:**
   The `[CLS]` token embedding (representing the entire narrative flow) is passed through a dense MLP (Linear -> ReLU -> Dropout -> Linear) to output a continuous plausibility score. We explicitly remove sigmoid bounding to prevent gradient saturation.

3. **Multi-Objective Loss Engineering:**
   To combat the unique nature of predicting continuous scores for ordinal data, our custom PyTorch training loop minimizes a sum of three distinct penalties:
   * **Huber Loss:** For robust absolute score prediction and outlier penalization.
   * **RankNet Pairwise Loss:** To explicitly penalize the model for incorrectly ordering highly plausible vs. implausible stories in a mini-batch, directly optimizing for Spearman correlation.
   * **Uncertainty-Aware Margin Loss:** Utilizes the human standard deviation (`stdev`) as a tolerance boundary. Loss is zero if the prediction falls within natural human disagreement, penalizing linearly otherwise.

4. **Nelder-Mead Threshold Optimization:**
   Post-training, we use `scipy.optimize.minimize` (Nelder-Mead method) to dynamically adjust the classification boundaries (from the default 1.5, 2.5, 3.5, 4.5) to mathematically maximize the discrete Macro F1 score on the validation set.

## Experimental Results

Our final model achieved the following performance on the validation set:
* **Accuracy at Standard Deviation:** ~0.70+
* **Spearman Correlation:** ~0.50
* **Optimized Macro F1 Score:** ~0.31

The Nelder-Mead threshold optimization successfully shifted the classification boundaries to account for model biases, consistently improving our discrete F1 score by roughly 2 percent over default rounding.

## Challenges Faced and Discussion

1. **Shifting Architecture:** As mentioned, our initial literature review led us to build complex Hierarchical Dual Encoders. We quickly discovered that for short text, flat transformers massively outperform hierarchical models due to unrestricted self-attention.
2. **Overfitting to Training Data:** During training, our model began memorizing the training set by Epoch 10 (validation loss plateaued at ~1.0 while training loss plummeted to 0.23). We mitigated this using a strict Early Stopping mechanism that restored the best weights. In future iterations, we plan to increase Dropout to 0.3, raise Weight Decay to 0.1, and lower the regression head learning rate to prevent this gap entirely.
3. **Regression to the Mean:** Because we used a regression loss (Huber) to predict ordinal scores from 1 to 5, the model naturally became hesitant to guess the absolute extremes (1 and 5), favoring 2 and 4 to minimize massive error penalties. 
   * **Future Work:** To solve this structural issue, we plan to implement either Ordinal Regression (treating the scores as ordered binary categories) or Sample Weighting to artificially feed the model more extreme scores during training. Focal weighting was also tested but ultimately removed in favor of pure Huber loss to stabilize absolute Accuracy@std.

## Dataset Overview: AmbiStory
* **Training Set:** 2,280 entries
* **Validation Set:** 588 entries
* **Test Set:** Strictly withheld during development to prevent data leakage.

## Directory Structure

```text
├── data/                      <- dataset folder (JSON files)
├── notebooks/                 <- Jupyter notebooks
│   ├── EDA.ipynb              <- Exploratory Data Analysis
│   ├── WSD_Final_Colab.ipynb  <- Testing playground and old architecture logs
│   └── Group02_final_code.ipynb <- Final submission notebook with Nelder-Mead optimization
├── src/                       <- Source code modules
│   ├── data_loader.py         <- Data parsing, splitting, and tokenization
│   ├── feature_extraction.py  <- DeBERTa initialization
│   ├── custom_losses.py       <- Multi-objective PyTorch loss classes
│   └── evaluation.py          <- Metrics (Acc@std, Spearman, F1, Confusion Matrix)
├── README.md                  <- Project documentation
└── requirements.txt           <- Required Python packages
```

## Team Members & Roles

We have divided the workload into distinct, parallel tracks:

| Team Member | Responsibilities |
| :--- | :--- |
| **Raid Bahadir** | Finalize JSON parsing, ensure strict train/dev splits, and manage tokenization logic for independent narrative strings. |
| **Areeb Kamal** | Program the PyTorch BiLSTM layer, manage hidden state concatenation with the target meaning, and construct the final MLP regression head. |
| **Ramzy Dahhani** | Code the custom RankNet and Uncertainty-Aware loss classes from scratch in PyTorch, and manage lambda weights in the main backward pass. |
| **Hamza Uysal** | Build automated scoring scripts for SemEval metrics (Acc@std, Spearman), write continuous-to-discrete rounding algorithms, and generate confusion matrices/F1-scores. |
| **Moeed ur Rehman** | Directed the complete architectural shift to the Flat DeBERTa-v3-base framework. Implemented the final pipeline including flat text concatenation, regression head construction, data loader dictionary mapping fixes, integration of the Multi-Objective Loss, threshold optimization via Nelder-Mead, and final model compilation. |

---
*Created for CS445: Natural Language Processing. Sabanci University.*
