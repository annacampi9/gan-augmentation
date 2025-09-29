import re
import matplotlib.pyplot as plt

# ---------------------------
# 1. Paths to refinement log files
# ---------------------------
files = {
    "B2 Mixup": "logs/B2_Mixup_a0p5_start220_ls0p1_ema0p999.txt",
    "B2.1": "logs/b2.1.txt",
    "B2.2": "logs/b2.2.txt",
}

# Regex to capture epoch, test acc, and test F1
pattern = re.compile(r"Epoch\s+(\d+).*?test acc=([\d.]+).*?test F1=([\d.]+)")

# Parse results
results = {}
for label, path in files.items():
    epochs, accs, f1s = [], [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epoch = int(m.group(1))
                acc = float(m.group(2))
                f1 = float(m.group(3))
                epochs.append(epoch)
                accs.append(acc)
                f1s.append(f1)
    results[label] = {"epochs": epochs, "acc": accs, "f1": f1s}

# ---------------------------
# 2. Accuracy curves
# ---------------------------
# plt.figure(figsize=(10,6))
# for label, data in results.items():
#     plt.plot(data["epochs"], data["acc"], label=label)
# plt.xlabel("Epochs")
# plt.ylabel("Test Accuracy")
# plt.title("Test Accuracy vs Epochs (B2, B2.1, B2.2)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# ---------------------------
# 3. F1-score curves
# ---------------------------
plt.figure(figsize=(10,6))
for label, data in results.items():
    plt.plot(data["epochs"], data["f1"], label=label)
plt.xlabel("Epochs")
plt.ylabel("Test F1-score")
plt.title("Test F1-score vs Epochs (B2, B2.1, B2.2)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
