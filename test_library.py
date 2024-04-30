import json
from library import LPTooling

with open("lp1.json", "r" ,encoding="utf-8") as f:
    data = json.load(f)

sentences = [s["prompt"] for s in data["data"]][:10]

tool = LPTooling(sentences, "microsoft/Phi-3-mini-128k-instruct", "phi-3-alpaca_instruct-tuned-k2000")
tool.compute_perplexities(0)
tool.compute_perplexities(1)
tool.save("my_save.json")