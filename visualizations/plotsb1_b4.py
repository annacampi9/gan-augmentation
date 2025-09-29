import re
import matplotlib.pyplot as plt

# ---------------------------
# 1. Paths to your log files
# ---------------------------
files = {
    "B1 CutMix": "logs/B1_CutMix_a2p0_p0p7_start260_ls0p1_ema0p999.txt",
    "B2 Mixup": "logs/B2_Mixup_a0p5_start220_ls0p1_ema0p999.txt",
    "B3 Mixup+CutMix": "logs/B3_MixupCutMix_a0p8_p0p6_start220_ls0p05_ema0p999.txt",
    "B4 Transfer": "logs/B4_transferlearning.txt",
}

# ---------------------------
# 2. Regex to extract data
# ---------------------------
pattern = re.compile(r"Epoch\s+(\d+).*?test acc=([\d.]+).*?test F1=([\d.]+)")

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
# 3. Accuracy curves
# ---------------------------
# plt.figure(figsize=(10,6))
# for label, data in results.items():
#     plt.plot(data["epochs"], data["acc"], label=label)
# plt.xlabel("Epochs")
# plt.ylabel("Test Accuracy")
# plt.title("Test Accuracy vs Epochs (B1–B4)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# ---------------------------
# 4. F1-score curves
# ---------------------------
# plt.figure(figsize=(10,6))
# for label, data in results.items():
#     plt.plot(data["epochs"], data["f1"], label=label)
# plt.xlabel("Epochs")
# plt.ylabel("Test F1-score")
# plt.title("Test F1-score vs Epochs (B1–B4)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# ---------------------------
# 5. Zoomed-in B4 only
# ---------------------------
plt.figure(figsize=(10,6))
plt.plot(results["B4 Transfer"]["epochs"], results["B4 Transfer"]["acc"], label="Accuracy")
plt.plot(results["B4 Transfer"]["epochs"], results["B4 Transfer"]["f1"], label="F1-score")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("Zoomed-in view of B4 Transfer Learning (flat performance)")
plt.legend()
plt.grid(True)
plt.ylim(0, 0.2)   # zoom y-axis for clarity
plt.tight_layout()
plt.show()
