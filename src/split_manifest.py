import json
from sklearn.model_selection import train_test_split

manifest_path = "/home/giolinux/NLP_Project/audio/greta1/data_manifest.json"
train_manifest = "train_manifest.json"
val_manifest = "val_manifest.json"

with open(manifest_path, 'r') as f:
    lines = f.readlines()

train_lines, val_lines = train_test_split(lines, test_size=0.2, random_state=42)

with open(train_manifest, 'w') as f:
    f.writelines(train_lines)

with open(val_manifest, 'w') as f:
    f.writelines(val_lines)

print("Data split into training and validation sets.")
