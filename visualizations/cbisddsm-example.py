import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# ---------------------------
# Paths
# ---------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, "data")
csv_dir = os.path.join(base_dir, "csv")
jpeg_dir = os.path.join(base_dir, "jpeg")

# Load metadata
calc_df = pd.read_csv(os.path.join(csv_dir, "calc_case_description_train_set.csv"))
mass_df = pd.read_csv(os.path.join(csv_dir, "mass_case_description_train_set.csv"))

# ---------------------------
# Helper to load a single image
# ---------------------------
def load_image(row, size=(256, 256)):
    rel_path = row["image file path"]
    parts = rel_path.split("/", 1)
    if len(parts) > 1:
        rel_path = parts[1]

    uid = os.path.basename(os.path.dirname(rel_path))
    img_folder = os.path.join(jpeg_dir, uid)

    if os.path.exists(img_folder):
        jpgs = [f for f in os.listdir(img_folder) if f.lower().endswith(".jpg")]
        if jpgs:
            img_path = os.path.join(img_folder, jpgs[0])
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                return cv2.resize(image, size)
    return None

def get_one_example(df, pathology):
    row = df[df["pathology"].str.contains(pathology)].iloc[0]
    return load_image(row)

# ---------------------------
# Collect 1 example per class
# ---------------------------
calc_benign = get_one_example(calc_df, "BENIGN")
calc_malignant = get_one_example(calc_df, "MALIGNANT")
mass_benign = get_one_example(mass_df, "BENIGN")
mass_malignant = get_one_example(mass_df, "MALIGNANT")

# ---------------------------
# Plot 2x2 grid
# ---------------------------
fig, axes = plt.subplots(2, 2, figsize=(6, 6))

images = [
    (calc_benign, "Calcification - Benign"),
    (calc_malignant, "Calcification - Malignant"),
    (mass_benign, "Mass - Benign"),
    (mass_malignant, "Mass - Malignant"),
]

for ax, (img, title) in zip(axes.ravel(), images):
    if img is not None:
        ax.imshow(img, cmap="gray")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")

plt.tight_layout()
plt.show()

# Save for thesis
# plt.savefig("cbis_examples_2x2.png", dpi=300, bbox_inches="tight")
