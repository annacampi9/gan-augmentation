import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Example confusion matrix (replace with your own values)
cm = np.array([
    [732,  40,  65,  19,   5,   7,   5,  14,  87,  26],
 [ 17, 869,  12,   3,   2,  11,   4,   6,  20,  56],
 [ 98,  13, 591,  72,  48,  87,  33,  26,  20,  12],
 [ 34,  10,  78, 469,  32, 250,  50,  22,  31,  24],
 [ 50,  16, 151, 100, 414, 109,  29, 101,  25,   5],
 [ 14,   3,  69, 178,  26, 629,  12,  44,  12,  13],
 [ 20,  28,  96,  96,  30,  85, 605,  10,  20,  10],
 [ 27,  10,  49,  61,  46, 115,   4, 655,   8,  25],
 [ 97,  37,  18,  23,   1,  15,   3,   7, 783,  16],
 [ 27, 108,  13,  20,   3,  14,   4,  12,  53, 746]
])

labels = ["airplane","auto","bird","cat","deer","dog","frog","horse","ship","truck"]

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (B2, CIFAR-10)")
plt.tight_layout()
plt.show()
