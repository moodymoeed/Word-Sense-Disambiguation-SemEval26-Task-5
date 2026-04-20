# CS445 Group 2: Word Sense Disambiguation (SemEval-26 Task 5)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-F9D371.svg)](https://huggingface.co/)

**Official repository for CS445 Group 2. We tackle SemEval-2026 Task 5 (AmbiStory) by predicting continuous word sense plausibility using a hierarchical LoRA-adapted DeBERTa + BiLSTM architecture with multi-objective loss.**

---

##  Project Overview

Traditional Word Sense Disambiguation (WSD) treats language as a multiple-choice test. However, in real-world contexts, multiple interpretations of a word can remain plausible simultaneously. **SemEval-2026 Task 5** models this phenomenon using the **AmbiStory dataset**, requiring systems to predict human plausibility ratings (on a continuous scale from 1 to 5) for candidate word senses given a 5-sentence narrative.

Instead of relying on closed-source LLM APIs, our group proposes a novel, self-contained deep learning methodology. We explicitly step away from the standard procedure of flat-concatenating narratives. Instead, we propose a **Unified Hierarchical Component Encoding with Multi-Objective Loss** to capture both deep semantic relationships and chronological narrative flow.

##  System Architecture

Our system pipeline consists of four main stages:

1. **Disentangled Feature Extraction (DeBERTa + LoRA):** 
   Instead of flattening the story, we parse it into four distinct components: `precontext`, `ambiguous_sentence`, `ending`, and `judged_meaning`. Each string is encoded independently using `microsoft/deberta-large`. To manage GPU memory constraints on undergraduate hardware, we freeze the base model and inject low-rank decomposition matrices into the attention layers using Low-Rank Adaptation (**LoRA**).
   
2. **Temporal Sequence Modeling (BiLSTM):**
   The extracted dense embeddings for the narrative components (`precontext` -> `sentence` -> `ending`) are fed sequentially into a PyTorch Bidirectional LSTM. This captures the retroactive contextualization inherent in the dataset (i.e., how the ending changes the interpretation of the setup).
   
3. **Dense Regression Head:**
   The final hidden state of the BiLSTM (representing the entire temporal narrative flow) is mathematically concatenated with the independently encoded `judged_meaning` vector. This combined representation is passed through a dense MLP to output a continuous plausibility score (1.0 to 5.0).

4. **Multi-Objective Loss Engineering:**
   To combat the U-shaped distribution of the human annotator ratings, our custom PyTorch training loop minimizes a sum of three distinct penalties:
   * **Huber Loss:** For robust absolute score prediction and outlier penalization.
   * **RankNet Pairwise Loss:** To explicitly penalize the model for incorrectly ordering highly plausible vs. implausible stories in a mini-batch (directly optimizing for Spearman correlation).
   * **Uncertainty-Aware Margin Loss:** Utilizes the human standard deviation (`stdev`) as a tolerance boundary. Loss is zero if the prediction falls within natural human disagreement, penalizing linearly otherwise.

##  Dataset: AmbiStory
* **Training Set:** 2,280 entries (79.5%)
* **Validation Set:** 588 entries (20.5%)
* **Test Set:** Strictly withheld during development to prevent data leakage.

##  Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/moodymoeed/Word-Sense-Disambiguation-SemEval26-Task-5.git
   cd Word-Sense-Disambiguation-SemEval26-Task-5
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

##  Directory Structure

```text
├── data/                      <- dataset folder (JSON files)
├── notebooks/                 <- Jupyter notebooks
│   ├── EDA.ipynb              <- Exploratory Data Analysis
│   └── Group02_final_code.ipynb <- Final submission notebook with retained output logs
├── src/                       <- Source code modules
│   ├── data_loader.py         <- Data parsing, splitting, and tokenization
│   ├── feature_extraction.py  <- DeBERTa initialization, LoRA config, and hierarchical forward pass
│   ├── sequence_model.py      <- BiLSTM temporal reasoning and regression head
│   ├── custom_losses.py       <- Multi-objective PyTorch loss classes
│   ├── evaluation.py          <- Metrics (Acc@std, Spearman, F1, Confusion Matrix)
│   └── train.py               <- Main training loop and backpropagation execution
├── README.md                  <- Project documentation
└── requirements.txt           <- Required Python packages
```

##  Team Members & Roles

We have divided the workload into five distinct, parallel tracks:

| Team Member | Role | Responsibilities |
| :--- | :--- | :--- |
| **Raid Bahadir** | Data Engineering & Pipeline Lead | Finalize JSON parsing, ensure strict train/dev splits, and manage tokenization logic for independent narrative strings. |
| **Moeed ur Rehman** | Transformer Fine-Tuning Lead | Initialize HuggingFace `DeBERTa-large`, configure PEFT/LoRA matrices for memory constraints, and build the hierarchical feature extraction forward-pass. |
| **Areeb Kamal** | Sequence Modeling Lead | Program the PyTorch BiLSTM layer, manage hidden state concatenation with the target meaning, and construct the final MLP regression head. |
| **Ramzy Dahhani** | Mathematical Optimization Lead | Code the custom RankNet and Uncertainty-Aware loss classes from scratch in PyTorch, and manage lambda weights in the main backward pass. |
| **Hamza Uysal** | Evaluation & ML Ops Lead | Build automated scoring scripts for SemEval metrics (Acc@std, Spearman), write continuous-to-discrete rounding algorithms, and generate confusion matrices/F1-scores. |

##  Deliverables & Usage
For our final submission, the complete working system, hyperparameter configurations, and training/evaluation executions will be compiled into a single, well-commented Jupyter Notebook (`Group02_final_code.ipynb`) located in the `notebooks/` directory.

---
*Created for CS445: Natural Language Processing. Sabanci University.*
