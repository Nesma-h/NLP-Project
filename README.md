# 📄 Text Summarization — Extractive & Abstractive Approaches

A structured NLP pipeline that compares two text summarization techniques — **TF-IDF extractive** summarization and **BART abstractive** summarization — with a full-featured **Streamlit web app** for interactive use.

---

## 🗂️ Project Structure

```
├── App.py                    # Streamlit web application
├── Text_Summarization.ipynb  # Full research & evaluation notebook
└── data.csv                  # Dataset (Content + Summary columns)
```

---

## ✨ Features

- **Two summarization methods** available side by side:
  - 📌 **Extractive (TF-IDF)** — ranks sentences by importance and returns the top-N in original order; fast, lightweight, no model download required.
  - 🧠 **Abstractive (BART)** — uses `facebook/bart-large-cnn`, a transformer fine-tuned on news summarization, to generate fluent paraphrased summaries.
- **Adjustable output length** — choose 1–10 sentences.
- **Live statistics** — word count, sentence count, character count, and compression ratio.
- **Jupyter notebook** — end-to-end pipeline covering data loading, EDA, preprocessing, model evaluation, and ROUGE scoring with visualizations.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/text-summarization.git
cd text-summarization
```

### 2. Install dependencies

> ⚠️ PyTorch must be installed **before** `transformers`. Use the CPU-only build for broad compatibility.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers scikit-learn nltk pandas numpy streamlit matplotlib seaborn textstat contractions tqdm
```

### 3. Run the Streamlit app

```bash
streamlit run App.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📓 Notebook Walkthrough

Open `Text_Summarization.ipynb` in Jupyter and run cells in order:

| Part | Description |
|------|-------------|
| **1.0** | Dependency installation — run once, then restart the kernel |
| **1.1** | Environment verification (PyTorch check) |
| **1.2** | Imports |
| **1.3** | Data loading (50,000 records sample) |
| **1.4** | Data exploration & cleaning |
| **1.5** | Exploratory Data Analysis (EDA) |
| **2** | Text preprocessing pipeline |
| **3** | TF-IDF extractive summarization |
| **4** | BART abstractive summarization |
| **5** | ROUGE evaluation & comparison charts |

> ⏱️ **Note:** The first BART run downloads ~1.6 GB of model weights. Subsequent runs use the cached version.

---

## 📊 Evaluation

Model performance is measured using **ROUGE** scores (ROUGE-1, ROUGE-2, ROUGE-L) against reference summaries. Results are visualized as a bar chart comparing TF-IDF vs. BART across all metrics.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `transformers` | BART model (`facebook/bart-large-cnn`) |
| `torch` | PyTorch backend for BART inference |
| `scikit-learn` | TF-IDF vectorization |
| `nltk` | Sentence tokenization, stopwords |
| `streamlit` | Interactive web application |
| `pandas` / `numpy` | Data handling |
| `matplotlib` / `seaborn` | Visualizations |
| `rouge-score` | ROUGE evaluation metrics |

---

## 📋 Dataset

The dataset (`data.csv`) contains two columns:

| Column | Description |
|---|---|
| `Content` | Full article or document text |
| `Summary` | Reference human-written summary |

50,000 records are sampled for development and evaluation.

---

## 📸 App Preview

> The app features a dark-themed UI with a split layout — **Input** on the left, **Summary Output** on the right — with real-time statistics and a sidebar for method and length configuration.

---

## 📄 License

This project is released for educational purposes. Feel free to fork, modify, and build on it.
