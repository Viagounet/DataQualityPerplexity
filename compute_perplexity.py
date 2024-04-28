
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


perplexity = load("perplexity", module_type="metric")

dataset = load_dataset("llm-wizard/alpaca-gpt4-data")
new_data = {"prompt": []}

template =  """<|system|>
You are a helpful AI assistant.<|end|>
<|user|>
Input:
{{{input}}}
Instruction:
{{{instruction}}}<|end|>
<|assistant|>
{{{output}}}<|end|>"""

i = 0
for instruction, _input, _output in zip(dataset["train"]["instruction"], dataset["train"]["input"], dataset["train"]["output"]):
    i += 1
    if _input:
        prompt = template.replace("{{{input}}}", _input).replace("{{{instruction}}}", instruction).replace("{{{output}}}", _output)
    else:
        prompt = template.replace("Input:\n{{{input}}}", "").replace("{{{instruction}}}", instruction).replace("{{{output}}}", _output)
    new_data["prompt"].append(prompt)
    if i == 1000:
        break
ds = Dataset.from_dict(new_data)
ds = ds.train_test_split(test_size=0.15)

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "./phi-3-alpaca_instruct-tuned-k1000",
    device_map="auto",
    quantization_config=nf4_config,
    use_cache=False,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("./phi-3-alpaca_instruct-tuned-k1000")

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

perplexities_finetuned_model = []
for sentence in ds["train"]["prompt"][:20]:
    perplexity = calculate_perplexity(sentence, model, tokenizer)
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
for sentence in ds["train"]["prompt"][:20]:
    perplexity = calculate_perplexity(sentence, model, tokenizer)
    perplexities_base_model.append(perplexity)

for pp_ft, pp_bm in zip(perplexities_finetuned_model, perplexities_base_model):
    print(100*((pp_bm-pp_ft)/pp_bm))