import json
from library import LPTooling, tool_from_data, graph_length_vs_lp1

with open("data/prompts.json", "r" ,encoding="utf-8") as f:
    data = json.load(f)

sentences = [s["prompt"] for s in data["data"]]

tool = tool_from_data("data/prompts_lp1_residuals.json")

# tool = LPTooling(sentences, "microsoft/Phi-3-mini-128k-instruct", "phi-3-alpaca_instruct-tuned-k2000")
# tool.load_models()
# tool.compute_perplexities(0)
# tool.compute_perplexities(1)

# graph_length_vs_lp1([s.words for s in tool.sentences], [s.lp1 for s in tool.sentences], "graphs/length_vs_lp1.png")
tool.print_sentences(residual_range=[0.05, 0.1])
# tool.compute_residuals()
# tool.save("data/prompts_lp1_residuals.json")
