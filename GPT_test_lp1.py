
import torch
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
from trl import SFTTrainer
from transformers import TrainingArguments
from evaluate import load


nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "./phi-3-alpaca_instruct-tuned-k2000",
    device_map="auto",
    quantization_config=nf4_config,
    use_cache=False,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("./phi-3-alpaca_instruct-tuned-k2000")

def calculate_perplexity(sentence, model, tokenizer):
    # Load pre-trained model and tokenizer
    # Encode the sentence and add batch dimension
    inputs = tokenizer(sentence, return_tensors='pt')

    # Calculate log likelihood across the sequence
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        log_likelihood = outputs[0]  # This is the loss (negative log likelihood)
    # Calculate perplexity
    perplexity = torch.exp(log_likelihood)

    return perplexity.item()

with open("gpt_rephrasing.json", "r" ,encoding="utf-8") as f:
    data = json.load(f)

perplexities_finetuned_model = []
for sentence in data:
    perplexity = calculate_perplexity(sentence["gpt_prompt"], model, tokenizer)
    perplexities_finetuned_model.append(perplexity)


# print(f"The perplexity of the sentence is: {perplexity}")

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct",
    device_map="auto",
    quantization_config=nf4_config,
    use_cache=False,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

perplexities_base_model = []
for sentence in data:
    perplexity = calculate_perplexity(sentence["gpt_prompt"], model, tokenizer)
    perplexities_base_model.append(perplexity)

for sentence, pp_ft, pp_bm in zip(data, perplexities_finetuned_model, perplexities_base_model):
    lp_one = 100*((pp_bm-pp_ft)/pp_bm)
    sentence["gpt_lp1"] = lp_one
    with open("gpt_rephrasing_with_lp1.json", "w", encoding="utf-8") as f:
        json.dump({"data": data}, f, ensure_ascii=False, indent=4)