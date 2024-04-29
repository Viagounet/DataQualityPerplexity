
from datasets import load_dataset, Dataset
import json
import argparse
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from sentence_transformers import SentenceTransformer

dataset = load_dataset("llm-wizard/alpaca-gpt4-data")
new_data = {"prompt": []}



nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

phi_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct",
    device_map="auto",
    quantization_config=nf4_config,
    use_cache=False,
    trust_remote_code=True,
)
phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

with open("./lp1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def compute_embedding_phi(sentence):
    encoded_input = phi_tokenizer(sentence, return_tensors='pt')
    # Process the input through the model
    outputs = phi_model(**encoded_input, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    last_hidden_state = hidden_states[-1]  # Last hidden state
    last_token_embedding = last_hidden_state[:, -1, :]  # Embedding of the last token
    return last_token_embedding

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
def compute_embedding_miniLM(sentence):
    embeddings = model.encode([sentence])
    return embeddings

for sentence in data["data"]:
    # embedding = compute_embedding_phi(sentence)
    embedding_miniLM = compute_embedding_miniLM(sentence["prompt"])
    embedding_phi = compute_embedding_phi(sentence["prompt"])
    sentence["embedding_miniLM"] = embedding_miniLM.tolist()
    sentence["embedding_phi"] = embedding_phi.tolist()

with open("lp1_with_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)