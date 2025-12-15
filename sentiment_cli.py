#!/usr/bin/env python3
"""
Sentiment Analysis CLI with Explainable AI
"""
import argparse
import re
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, AutoModelForSequenceClassification
import json
import numpy as np
from sklearn.linear_model import Ridge
from scipy.special import comb
import warnings
warnings.filterwarnings('ignore')

# Load model
model_path = './best_roberta_model'
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained(model_path)

with open(f'{model_path}/label_mappings.json', 'r') as f:
    label_mappings = json.load(f)

id2label = {int(k): v for k, v in label_mappings['id2label'].items()}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
print(f"Model loaded on {device}\n")

def predict_sentiment_batch(texts):
    """Predict sentiment for batch of texts"""
    if isinstance(texts, str):
        texts = [texts]
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
    return probs.cpu().numpy()

# LIME Explainer
class LIMEExplainer:
    def __init__(self, predict_fn, num_samples=1000):
        self.predict_fn = predict_fn
        self.num_samples = num_samples

    def tokenize(self, text):
        # Simple word tokenization that separates punctuation
        # Split on whitespace and separate punctuation
        words = re.findall(r'\w+|[^\w\s]', text)
        return words

    def kernel_fn(self, distances):
        return np.sqrt(np.exp(-(distances ** 2) / 25 ** 2))

    def explain(self, text, target_class=None, verbose=True):
        words = self.tokenize(text)
        n_words = len(words)

        if verbose:
            print(f"  - Creating {self.num_samples} perturbations of {n_words} words...")
        perturbations = np.random.binomial(1, 0.5, (self.num_samples, n_words))
        perturbed_texts = [' '.join([w for w, m in zip(words, mask) if m == 1]) or ''
                          for mask in perturbations]

        if verbose:
            print(f"  - Running {self.num_samples} predictions...")
        predictions = self.predict_fn(perturbed_texts)

        if target_class is None:
            target_class = np.argmax(self.predict_fn([text])[0])

        if verbose:
            print(f"  - Computing importance scores...")
        y = predictions[:, target_class]
        distances = np.sum(1 - perturbations, axis=1)
        weights = self.kernel_fn(distances)

        ridge = Ridge(alpha=1.0)
        ridge.fit(perturbations, y, sample_weight=weights)

        importance = list(zip(words, ridge.coef_))
        importance.sort(key=lambda x: abs(x[1]), reverse=True)

        if verbose:
            print(f"  - Done!")
        return importance, target_class

# SHAP Explainer
class SHAPExplainer:
    def __init__(self, predict_fn, num_samples=500):
        self.predict_fn = predict_fn
        self.num_samples = num_samples

    def shapley_kernel(self, n, s):
        if s == 0 or s == n:
            return 1e10
        return (n - 1) / (comb(n, s) * s * (n - s))

    def explain(self, text, target_class=None, verbose=True):
        words = text.split()
        n_words = len(words)

        if target_class is None:
            target_class = np.argmax(self.predict_fn([text])[0])

        if verbose:
            print(f"  - Computing Shapley values ({self.num_samples} samples × {n_words} words = {self.num_samples * n_words * 2} predictions)...")
        empty_pred = self.predict_fn([''])[0, target_class]
        shap_values = np.zeros(n_words)

        for idx in range(self.num_samples):
            if verbose and idx > 0 and idx % 100 == 0:
                print(f"    Progress: {idx}/{self.num_samples} samples ({idx/self.num_samples*100:.1f}%)")
            z = np.random.binomial(1, 0.5, n_words)
            for i in range(n_words):
                z_with, z_without = z.copy(), z.copy()
                z_with[i], z_without[i] = 1, 0

                text_with = ' '.join([w for w, m in zip(words, z_with) if m == 1])
                text_without = ' '.join([w for w, m in zip(words, z_without) if m == 1])

                pred_with = self.predict_fn([text_with])[0, target_class] if text_with else empty_pred
                pred_without = self.predict_fn([text_without])[0, target_class] if text_without else empty_pred

                contrib = pred_with - pred_without
                weight = self.shapley_kernel(n_words, np.sum(z_without))
                shap_values[i] += contrib * weight

        shap_values /= self.num_samples
        importance = list(zip(words, shap_values))
        importance.sort(key=lambda x: abs(x[1]), reverse=True)

        if verbose:
            print(f"  - Done!")
        return importance, target_class

# Integrated Gradients Explainer
class IntegratedGradientsExplainer:
    def __init__(self, model, tokenizer, device, n_steps=50):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_steps = n_steps

    def explain(self, text, target_class=None, verbose=True):
        if verbose:
            print(f"  - Tokenizing and preparing embeddings...")
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=64)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        if target_class is None:
            with torch.no_grad():
                target_class = self.model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax().item()

        embeddings = self.model.roberta.embeddings(input_ids)
        embeddings.requires_grad_(True)
        baseline = torch.zeros_like(embeddings)

        if verbose:
            print(f"  - Computing gradients ({self.n_steps + 1} steps)...")
        gradients = []
        for i in range(self.n_steps + 1):
            if verbose and i > 0 and i % 10 == 0:
                print(f"    Progress: {i}/{self.n_steps + 1} steps ({i/(self.n_steps + 1)*100:.1f}%)")
            scaled_input = baseline + (float(i) / self.n_steps) * (embeddings - baseline)
            scaled_input.requires_grad_(True)

            outputs = self.model(inputs_embeds=scaled_input, attention_mask=attention_mask)
            score = outputs.logits[0, target_class]
            self.model.zero_grad()
            score.backward(retain_graph=True)
            gradients.append(scaled_input.grad.clone())

        if verbose:
            print(f"  - Aggregating gradients and computing attributions...")
        avg_gradients = torch.stack(gradients).mean(dim=0)
        integrated_gradients = (embeddings - baseline) * avg_gradients
        attributions = integrated_gradients.sum(dim=-1).squeeze(0).cpu().detach().numpy()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        importance = [(t, a) for t, a in zip(tokens, attributions) if t not in ['<s>', '</s>', '<pad>']]
        importance.sort(key=lambda x: abs(x[1]), reverse=True)

        if verbose:
            print(f"  - Done!")
        return importance, target_class

def print_separator():
    print("=" * 80)

def print_explanation(importance, method_name, top_k=10):
    print(f"\n{method_name} Explanation (Top {top_k} Features):")
    print("-" * 80)
    for i, (word, score) in enumerate(importance[:top_k], 1):
        direction = "+" if score > 0 else "-"
        print(f"{i:2d}. {word:20s} {direction} {abs(score):8.4f}")

def analyze_sentiment(text, method='all', top_k=10):
    print_separator()
    print(f"TEXT: {text}")
    print_separator()

    # Get prediction
    probs = predict_sentiment_batch(text)[0]
    pred_class = np.argmax(probs)
    pred_label = id2label[pred_class]
    confidence = probs[pred_class]

    print(f"\nPREDICTION: {pred_label}")
    print(f"CONFIDENCE: {confidence:.2%}")

    print("\nAll Class Probabilities:")
    for i, prob in enumerate(probs):
        print(f"  {id2label[i]:20s}: {prob:.4f}")

    # Get explanations
    if method in ['lime', 'all']:
        print("\nGenerating LIME explanation...")
        lime_explainer = LIMEExplainer(predict_sentiment_batch, num_samples=1000)
        lime_imp, _ = lime_explainer.explain(text, target_class=pred_class)
        print_explanation(lime_imp, "LIME", top_k)

    if method in ['shap', 'all']:
        print("\nGenerating SHAP explanation...")
        shap_explainer = SHAPExplainer(predict_sentiment_batch, num_samples=500)
        shap_imp, _ = shap_explainer.explain(text, target_class=pred_class)
        print_explanation(shap_imp, "SHAP", top_k)

    if method in ['ig', 'all']:
        print("\nGenerating Integrated Gradients explanation...")
        ig_explainer = IntegratedGradientsExplainer(model, tokenizer, device, n_steps=50)
        ig_imp, _ = ig_explainer.explain(text, target_class=pred_class)
        print_explanation(ig_imp, "Integrated Gradients", top_k)

    print_separator()

def main():
    parser = argparse.ArgumentParser(
        description='Sentiment Analysis with Explainable AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sentiment_cli.py "This movie is fantastic!"
  python sentiment_cli.py "Terrible experience" --method lime
  python sentiment_cli.py "It's okay" --method shap --top 5
  python sentiment_cli.py --interactive
        """
    )

    parser.add_argument('text', nargs='?', help='Text to analyze')
    parser.add_argument('-m', '--method',
                       choices=['lime', 'shap', 'ig', 'all'],
                       default='all',
                       help='XAI method to use (default: all)')
    parser.add_argument('-t', '--top', type=int, default=10,
                       help='Number of top features to show (default: 10)')
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='Interactive mode')

    args = parser.parse_args()

    if args.interactive:
        print("Interactive Sentiment Analysis (type 'quit' to exit)")
        print_separator()
        while True:
            try:
                text = input("\nEnter text: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                if not text:
                    continue
                analyze_sentiment(text, args.method, args.top)
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    elif args.text:
        analyze_sentiment(args.text, args.method, args.top)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
