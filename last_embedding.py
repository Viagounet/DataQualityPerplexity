
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

dataset = load_dataset("llm-wizard/alpaca-gpt4-data")
new_data = {"prompt": []}



nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct",
    device_map="auto",
    quantization_config=nf4_config,
    use_cache=False,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

with open("./lp1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for sentence in data["data"]:
    encoded_input = tokenizer(sentence["prompt"], return_tensors='pt')
    # Process the input through the model
    outputs = model(**encoded_input, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    last_hidden_state = hidden_states[-1]  # Last hidden state
    last_token_embedding = last_hidden_state[:, -1, :]  # Embedding of the last token
    sentence["embedding"] = last_token_embedding.tolist()

with open("lp1_with_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)