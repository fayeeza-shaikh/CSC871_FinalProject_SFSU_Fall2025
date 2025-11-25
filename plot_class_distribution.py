# %%
import os
import matplotlib.pyplot as plt

# Define your actual data path
base_path = r"C:\Users\91638\Desktop\CSC871_FinalProject_SFSU_Fall2025-main\CSC871_FinalProject_SFSU_Fall2025-main\chest_xray"

splits = ['train', 'val_new', 'test']
classes = ['NORMAL', 'PNEUMONIA']
counts = {split: {cls: 0 for cls in classes} for split in splits}

# Count number of images
for split in splits:
    for cls in classes:
        folder = os.path.join(base_path, split, cls)
        if os.path.exists(folder):
            counts[split][cls] = len(os.listdir(folder))
        else:
            print(f"⚠️ Folder not found: {folder}")

# Plot
x = range(len(splits))
bar_width = 0.35

normal_counts = [counts[split]['NORMAL'] for split in splits]
pneumonia_counts = [counts[split]['PNEUMONIA'] for split in splits]

plt.figure(figsize=(8,6))
plt.bar([i - bar_width/2 for i in x], normal_counts, bar_width, label='NORMAL')
plt.bar([i + bar_width/2 for i in x], pneumonia_counts, bar_width, label='PNEUMONIA')

plt.xticks(x, ['Train', 'Val', 'Test'])
plt.ylabel("Number of Images")
plt.title("Class Distribution per Dataset Split")
plt.legend()
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.show()



