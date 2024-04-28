import json
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import coolwarm  # This provides a blue-to-red colormap

with open("./lp1_with_embeddings_miniLM.json", "r", encoding="utf-8") as f:
    data = json.load(f)

tensor_data = []
learning_percentages = []

for data in data["data"]:
    tensor_data.append(data["embedding"][0])
    learning_percentages.append(data["learning_percentage"])


# Convert list of arrays to a single NumPy array
tensor_array = np.array(tensor_data)

# Compute t-SNE embedding
tsne = TSNE(n_components=2, random_state=0)
tsne_results = tsne.fit_transform(tensor_array)

# Normalizer for the learning percentages
norm = Normalize(vmin=min(learning_percentages), vmax=max(learning_percentages))

# Plotting
plt.figure(figsize=(8, 6))
sc = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=learning_percentages, cmap=coolwarm, norm=norm)

# Colorbar to show learning percentage gradient
cbar = plt.colorbar(sc)
cbar.set_label('Learning Percentage')

plt.title('t-SNE Visualization of Tensor Data')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

# Save the plot as a PNG file
plt.savefig('tsne_visualization_miniLM.png')

# Close the plot
plt.close()