
import json
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from sentence_transformers import SentenceTransformer


class Sentence:
    def __init__(self, content):
        self.content = content
        self.p0 = None
        self.p1 = None

    @property
    def lp1(self):
        return (self.p0 - self.p1) / self.p0

    def json(self):
        return {"content": self.content, "p0": self.p0, "p1": self.p1, "lp1": self.lp1}

def tool_from_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = f.read(path)
    sentences = []
    for sentence in data["sentences"]:
        s = Sentence(sentence["content"])
        for key, value in sentence.items():
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

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            quantization_config=nf4_config,
            use_cache=False,
            trust_remote_code=True,
        )
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        self.finetuned_model = AutoModelForCausalLM.from_pretrained(
            finetuned_model,
            device_map="auto",
            quantization_config=nf4_config,
            use_cache=False,
            trust_remote_code=True,
        )
        self.finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_model)


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
            print(sentence.p0)

    def save(self, path: str):
        sentences_dict = [s.json() for s in self.sentences]
        output_dict = {"base_model": self.base_model_str, "finetuned_model": self.finetuned_model_str, "sentences": sentences_dict}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, indent=4, ensure_ascii=False)