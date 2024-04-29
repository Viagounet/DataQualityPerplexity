from doc_llm.engine import Engine
import json
import matplotlib.pyplot as plt

augmented_dataset = []

engine = Engine("gpt-4-turbo")
# engine.query("Hello!!!")
with open("lp1.json", "r" ,encoding="utf-8") as f:
    data = json.load(f)

short_yet_hard_prompts = []

lengths = []
lps = []
for sentence in data["data"]:
    if sentence["learning_percentage"] == 0:
        continue
    lengths.append(sentence["prompt"].count(" "))
    lps.append(sentence["learning_percentage"])
    if sentence["learning_percentage"] < 70 and sentence["prompt"].count(" ") < 100:
        short_yet_hard_prompts.append(sentence)


title = "LP(1) vs Number of words"
plt.figure(figsize=(10, 6))
plt.scatter(lengths, lps, color='blue', alpha=0.5)  # Scatter plot
plt.title(title)
plt.xlabel('Number of words')
plt.ylabel('LP(1)')
plt.grid(True)

# Save the plot as a PNG file
plt.savefig(f'length_vs_learning_percentage.png')

# Close the plot
plt.close()

with_synthetic_data = []
for sentence in short_yet_hard_prompts:
    rephrasing_prompt = f"""<Instruction example>
{sentence['prompt']}
</Instruction example>

1. You will identify the themes of the instruction & answer.
2. You will propose a variation of this theme.
3. You will rewrite the instruction & the answer for the new theme.
IMPORTANT: You will follow the same format as the example. Meaning you'll start with <|system|> and end with <|end|>."""
    answer = engine.query(rephrasing_prompt, max_tokens=1024)
    with_synthetic_data.append({"original_prompt": sentence["prompt"], 
    "original_lp1": sentence["learning_percentage"], 
    "gpt_prompt": answer.content,
    "gpt_lp1": 0})
    with open("gpt_rephrasing.json", "w", encoding="utf-8") as f:
        json.dump(with_synthetic_data, f, ensure_ascii=False, indent=4)