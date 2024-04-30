
import json
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Tuple
import torch
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import math

class Sentence:
    def __init__(self, content):
        self.content = content
        self.p0 = None
        self.p1 = None
        self.residual = None

    @property
    def lp1(self):
        return (self.p0 - self.p1) / self.p0

    @property
    def words(self):
        return self.content.count(" ")

    def json(self):
        return {"content": self.content, "p0": self.p0, "p1": self.p1, "lp1": self.lp1, "words": self.words, "residual": self.residual}

def tool_from_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences = []
    for sentence in data["sentences"]:
        s = Sentence(sentence["content"])
        for key, value in sentence.items():
            if key in ["lp1", "words"]: #skipping lp1 as it's a property and not an attribute
                continue
            setattr(s, key, value)
        sentences.append(s)
    base_model = data["base_model"]
    finetuned_model = data["finetuned_model"]
    return LPTooling(sentences, base_model, finetuned_model)


class LPTooling:
    def __init__(self, sentences, base_model: str, finetuned_model: str):
        self.sentences = [Sentence(s) if type(s) is str else s for s in sentences]

        self.base_model_str = base_model
        self.finetuned_model_str = finetuned_model

    def load_models(self):
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_str,
            device_map="auto",
            quantization_config=nf4_config,
            use_cache=False,
            trust_remote_code=True,
        )
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_str)
        
        self.finetuned_model = AutoModelForCausalLM.from_pretrained(
            self.finetuned_model_str,
            device_map="auto",
            quantization_config=nf4_config,
            use_cache=False,
            trust_remote_code=True,
        )
        self.finetuned_tokenizer = AutoTokenizer.from_pretrained(self.finetuned_model_str)


    def compute_perplexities(self, i):
        if i == 0:
            current_model = self.base_model
            current_tokenizer = self.base_tokenizer
        else:
            current_model = self.finetuned_model
            current_tokenizer = self.finetuned_tokenizer

        for sentence in self.sentences:
            inputs = current_tokenizer(sentence.content, return_tensors='pt')
            # Calculate log likelihood across the sequence
            with torch.no_grad():
                outputs = current_model(**inputs, labels=inputs["input_ids"])
                log_likelihood = outputs[0]  # This is the loss (negative log likelihood)

            # Calculate perplexity
            perplexity = float(torch.exp(log_likelihood))
            setattr(sentence, f"p{i}", perplexity)

    def compute_residuals(self):
        lengths = np.array([s.words for s in self.sentences])
        lp1s = np.array([s.lp1 for s in self.sentences])
        params, _ = curve_fit(model, lengths, lp1s)
        lp1s_pred = model(np.array(lengths), *params)
        residuals = lp1s - lp1s_pred
        for sentence, residual in zip(self.sentences, residuals):
            sentence.residual = float(residual)

    def save(self, path: str):
        sentences_dict = [s.json() for s in self.sentences]
        output_dict = {"base_model": self.base_model_str, "finetuned_model": self.finetuned_model_str, "sentences": sentences_dict}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, indent=4, ensure_ascii=False)
    
    def print_sentences(self, lp_range=None, n_words=None, residual_range=None):
        sentences = [[s.content, s.lp1, s.words, s.residual] for s in self.sentences]
        sentences = sorted(sentences, key=lambda x: x[1])
        for sentence in sentences:
            if lp_range and not (lp_range[0] < sentence[1] < lp_range[1]):
                continue
            if n_words and not (n_words[0] < sentence[2] < n_words[1]):
                continue
            if residual_range and not (residual_range[0] < sentence[3] < residual_range[1]):
                continue
            print(f"LP(1) = {sentence[1]} | Nb words = {sentence[2]} | residual = {sentence[3]}")
            print(sentence[0])
            print("\n---------\n\n")


def model(y, a0, a1, a2, a3):
    return a0 + a1 * y + a2 * y**2 + a3 * y**3
    

def graph_length_vs_lp1(lengths: List[int], lp1s: List[float], output_path: str):
    title = "LP(1) vs Number of words"

    # Fit the model to the data
    params, _ = curve_fit(model, lengths, lp1s)

    # Generate data for the regression line
    lengths_fit = np.linspace(min(lengths), max(lengths), 300)
    lp1s_fit = model(lengths_fit, *params)

    plt.figure(figsize=(10, 6))
    plt.scatter(lengths, lp1s, color='blue', alpha=0.5)  # Scatter plot
    plt.plot(lengths_fit, lp1s_fit, color='red', label='Regression Curve')  # Regression curve
    plt.title(title)
    plt.xlabel('Number of words')
    plt.ylabel('LP(1)')
    plt.grid(True)
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig(output_path)

    # Close the plot
    plt.close()