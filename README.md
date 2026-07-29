# Multimodal Model Reliability Assessment

A multimodal framework for **post-deployment machine learning model reliability assessment**. The tool is model-agnostic, upload a deployed model with its baseline data, then upload production runs to diagnose why the model may be degrading. Results across five complementary analyses are summarized into a single **Fishbone (Ishikawa) diagram** and made queryable by a locally-served LLM.

---

## What it does

Production ML models drift as the data they see diverges from their training distribution. Most monitoring tools surface individual signals (feature drift, performance drop, prediction shift) in isolation, leaving practitioners to correlate dashboards by hand. This tool composes five analyses into one root-cause diagnosis:

1. **Performance Drift** — production metrics vs. benchmark thresholds set at upload time.
2. **Data Drift** — distribution shift in input features (KS, PSI, JS divergence, Chi-Square, combined by majority vote).
3. **Data Quality** — completeness, uniqueness, and validity of incoming production data.
4. **Prediction Drift** — shift in the model's output distribution (label shift).
5. **Model Explanations** — SHAP (structured data) and LIME (text/image) to surface attribution drift.

The findings roll up into a Fishbone diagram where each analysis is a "bone", so a user sees at a glance where the degradation is coming from.

## Key features

- **Modalities** — supports regression, classification, NLP (text classification, seq2seq summarization), and computer vision (image classification).
- **Fishbone root-cause summary** — one glanceable diagnostic per production run.
- **Natural-language query interface** — ask, in plain English, why a model is degrading; answers are grounded in the actual analysis results (retrieval-augmented), not the LLM's priors.
- **Subgroup filtering** — constrain all five analyses to a subpopulation (numeric range, category, or keyword) to catch localized drift that global tests miss.
- **Cached diagnostics** — expensive computation runs once at production-run upload; every later view reads from cached results.

## Architecture

The tool is a four-layer application:

| Layer | Implementation |
| --- | --- |
| **Presentation** | Multi-page Streamlit UI (`1_Landing_Page.py`, `pages/`) |
| **State** | JSON files — `models.json` (registry), `model_info.json` (per-model metadata + baseline stats), `results.json` (per-run analysis results) |
| **Compute** | `utils.py` — Implements every statistical test, drift detector, quality metric, feature extractor, and visualization |
| **LLM** | Ollama-served Mistral 7B, orchestrated via LangChain, with per-model prompt templates |

## Repository structure

```
model-reliability-assessment/
├── 1_Landing_Page.py      # Streamlit entry point (Fishbone + LLM query)
├── pages/                 # Add Model, Add Production Run, Analysis pages
├── utils.py               # Compute layer: statistical tests, drift detectors, metrics, viz
├── Jupyter Notebooks/     # Model training / experimentation notebooks
├── requirements.txt       # Python dependencies
└── README.md
```

## Getting started

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/) installed and running locally
- The Mistral 7B model pulled into Ollama

### Installation

```bash
# Clone the repository
git clone https://github.com/AkshatDasula/model-reliability-assessment.git
cd model-reliability-assessment

# (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Pull the LLM used by the query interface
ollama pull mistral
```

### Running the app

```bash
python3 -m streamlit run 1_Landing_Page.py
```

This launches the Streamlit interface in your browser.

## Usage

1. **Add a model** — register a new model by uploading baseline data and model artifacts, providing domain knowledge, and setting benchmark metrics. This also registers a per-model LLM template.
2. **Add a production run** — upload a new batch of production data with ground truths. All five analyses run and results are cached.
3. **Diagnosis** — select the model and run to see the Fishbone diagram, go to the *Analysis* page for per-analysis visualizations, or ask the LLM interface.

## Models evaluated

The framework was validated across five models spanning its four supported families:

| Model | Task | Dataset |
| --- | --- | --- |
| Medical Cost Prediction | Regression | Medical Cost Personal (Kaggle) |
| Heart Disease Classification | Binary classification | Heart Failure Prediction (Kaggle) |
| Bone Fracture Classification | Image classification | Bone Fracture X-Ray (Kaggle) |
| Sentiment Classification | Text classification | IMDB Reviews (Kaggle) |
| Text Summarization | Seq2seq generation | CNN/DailyMail (Kaggle) |

For structured-data models, controlled drift was induced by perturbing production data (mean/variance shift, outlier injection, category flips) at a tunable intensity. For text and image models, natural distributional variation between held-out batches was used.


## Acknowledgments

Developed as a data science capstone under the advisement of Prof. Jaideep Srivastava.
