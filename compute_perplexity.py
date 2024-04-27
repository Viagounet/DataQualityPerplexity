
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
print(ds)
s1 = ds["train"]["prompt"][0]
s2 = ds["train"]["prompt"][0]
s3 = ds["train"]["prompt"][0]

results = perplexity.compute(predictions=[s1, s2, s3], model_id='./phi-3-alpaca_instruct-tuned-k1000')
print(results)

results = perplexity.compute(predictions=[s1, s2, s3], model_id='microsoft/Phi-3-mini-128k-instruct')
print(results)