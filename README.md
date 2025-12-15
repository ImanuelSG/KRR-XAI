# KRR-XAI: Sentiment Analysis with Explainability

This project implements a sentiment analysis model using RoBERTa and provides explainability features using LIME, SHAP, and LRP (Layer-wise Relevance Propagation). It allows users to not only classify text sentiment but also understand *why* the model made a specific prediction.

## Features

*   **Robust Sentiment Classification**: Finetuned RoBERTa model for high-accuracy sentiment analysis.
*   **Explainable AI (XAI)**:
    *   **LIME** (Local Interpretable Model-agnostic Explanations): Perturbation-based explanation.
    *   **SHAP** (SHapley Additive exPlanations): Game theory-based feature importance.
    *   **LRP** (Layer-wise Relevance Propagation): Gradient-based relevance scoring.
*   **CLI Interface**: Easy-to-use command-line interface for analyzing text.
*   **Interactive Mode**: continuous analysis loop for testing multiple inputs.

## Installation

This project uses `pyproject.toml` for dependency management.

### Method 1: Using pip

```bash
pip install -e .
```

### Method 2: Using uv (Recommended for speed)

If you have [uv](https://github.com/astral-sh/uv) installed:

```bash
uv sync
```

## Usage

### 1. Training the Model

Before using the CLI, you need to train the model.

1.  Ensure you have your dataset in `dataset/train.csv`.
2.  Run the training notebook `train_roberta_optimized.ipynb`.
3.  This will save the fine-tuned model to `./best_roberta_model`.

### 2. Command-Line Interface (CLI)

Use `sentiment_cli.py` to analyze text.

**Basic Usage:**

```bash
python sentiment_cli.py "This movie is absolutely fantastic!"
```

**Specify Explainability Method:**

Evaluate using specific methods (default is `all`):

```bash
python sentiment_cli.py "The service was terrible but the food was okay." --method lime
python sentiment_cli.py "The service was terrible but the food was okay." --method shap --top 5
```

**Interactive Mode:**

```bash
python sentiment_cli.py --interactive
```

## Project Structure

*   `sentiment_cli.py`: Main CLI tool for inference and explanation.
*   `train_roberta_optimized.ipynb`: Notebook for training the RoBERTa model.
*   `XAI_Sentiment_Analysis_lib.ipynb` & `_scratch.ipynb`: Research and experimentation notebooks.
*   `pyproject.toml`: Project dependencies and configuration.

## Requirements

*   Python >= 3.9
*   Torch >= 2.0.0
*   Transformers >= 4.30.0
