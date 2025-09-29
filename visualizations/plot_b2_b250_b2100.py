import matplotlib.pyplot as plt

# Your B2 (Mixup) results
samples = [50, 100, 200]
acc_b2 = [42.7, 56.0, 66.0]

# Original DADA_augmented results
dada_samples = [50, 100, 200]
dada_acc = [46.0, 52.0, 62.0]

plt.figure(figsize=(6,5))

# Plot your results
plt.plot(samples, acc_b2, marker="o", color="blue", label="B2 (Mixup)")

# Plot DADA_augmented results
plt.plot(dada_samples, dada_acc, marker="s", color="red", linestyle="--", label="Original DADA_augmented")

# Formatting
plt.xlabel("Samples per Class")
plt.ylabel("Test Accuracy")
plt.title("Scaling Results: B2 vs. Original DADA_augmented (CIFAR-10)")
plt.xticks([50, 100, 200])
plt.ylim(35, 70)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig("scaling_comparison_with200.png", dpi=300)
plt.show()
