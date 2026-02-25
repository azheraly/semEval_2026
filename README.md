# Cross-Domain Detection of Machine-Generated Code via Stylometric Features and Transformer Models

The rise of generative models has made it increasingly difficult to distinguish machine-generated code from human-written code — especially across different programming languages, domains, and generation techniques. 

This Research focuses on classifying source code into two categories: -
**Human-written code** - **Machine-generated code** (e.g., GPT, Qwen,
DeepSeek, etc.) The objective is to identify stylistic, structural, and
statistical differences between human and machine-written code and train
machine learning models to classify them effectively.


### Subtask A: Binary Machine-Generated Code Detection

**Goal:**  
Given a code snippet, predict whether it is:

- **(i)** Fully **human-written**, or  
- **(ii)** Fully **machine-generated**

**Training Languages:** `C++`, `Python`, `Java`  
**Training Domain:** `Algorithmic` (e.g., Leetcode-style problems)

**Evaluation Settings:**

| Setting                              | Language                | Domain                 |
|--------------------------------------|-------------------------|------------------------|
| (i) Seen Languages & Seen Domains    | C++, Python, Java       | Algorithmic            |
| (ii) Unseen Languages & Seen Domains | Go, PHP, C#, C, JS      | Algorithmic            |
| (iii) Seen Languages & Unseen Domains| C++, Python, Java       | Research, Production   |
| (iv) Unseen Languages & Domains      | Go, PHP, C#, C, JS      | Research, Production   |

**Dataset Size**: 
- Train - 500K samples (238K Human-Written | 262K Machine-Generated)
- Validation - 100K samples

**Target Metric** - Macro F1-score (we will build the leaderboard based on it), but you are free to use whatever works best for your approach during training.

---


## 📁 Data Format

- All data will be released via:
  - [Kaggle](https://www.kaggle.com/competitions/sem-eval-2026-task-13-subtask-a)  
  - [HuggingFace Datasets](https://huggingface.co/datasets/DaniilOr/SemEval-2026-Task13)
  - In this GitHub repo as `.parquet` file

- For each subtask:
  - Dataset contains `code`,  `label` (which is label id), and additional meta-data such as programming language (`language`), and the `generator`.
  - Label mappings (`label_to_id.json` and `id_to_label.json`) are provided in each task folder  

---


##  Virtual Environment Setup

Below are instructions for Linux, macOS, and Windows.

###  Linux / macOS

**Create virtual environment**

``` bash
python3 -m venv venv
```

**Activate**

``` bash
source venv/bin/activate
```

**Deactivate**

``` bash
deactivate
```

### Windows (PowerShell)

**Create virtual environment**

``` powershell
python -m venv venv
```

**Activate**

``` powershell
venv\Scripts\Activate
```

**Deactivate**

``` powershell
deactivate
```

## 📦 Install Dependencies

``` bash
pip install -r requirements.txt
```

