import os

folders = [
    "data/raw/invoices",
    "data/raw/emails",
    "data/raw/contracts",
    "data/raw/news",
    "data/processed",
    "data/labeled"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"✅ Created {folder}")

print("\n✅ Setup complete! Now run 01_data_collection.ipynb")