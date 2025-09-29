import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Paths to the snapshots you want to show
files = [
    "samples_0000.png",
    "samples_0100.png",
    "samples_0150.png",
    "samples_0200.png"
]

epochs = [0, 100, 150, 200]

# Create a 2×2 grid
fig, axes = plt.subplots(2, 2, figsize=(7, 7))  # slightly smaller figure to keep things tight

for i, (file, epoch) in enumerate(zip(files, epochs)):
    row = i // 2
    col = i % 2
    img = mpimg.imread(file)
    axes[row, col].imshow(img)
    axes[row, col].set_title(f"Epoch {epoch}", fontsize=12)
    axes[row, col].axis("off")

# Adjust spacing: reduce horizontal and vertical gaps
plt.subplots_adjust(wspace=0.05, hspace=0.15)

plt.savefig("generator_progression_2x2.png", dpi=300, bbox_inches="tight")
plt.show()
