"""Load CSV with encoding=latin-1
keep only 'text' and 'sentiment' columns
drop the 1 row with missing text 
split
save as separate csvs"""

import pandas as pd
from sklearn.model_selection import train_test_split  # pyright: ignore[reportMissingModuleSource]

df = pd.read_csv("data/train.csv", encoding='latin-1')

dataset = df[['text', 'sentiment']]

dataset = dataset.dropna(subset=['text'])
print(f"Total samples after cleaning: {len(dataset)}")

dataset_sample = dataset.groupby('sentiment').sample(
    n=167,  # ~500 total examples
    random_state=42
)
print(f"Sampled: {len(dataset_sample)} examples")
print(dataset_sample['sentiment'].value_counts())

train, temp = train_test_split(dataset_sample, test_size=0.2, random_state=42, stratify=dataset_sample['sentiment'])

validate, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['sentiment'])

total = len(dataset_sample)  # or len(dataset)
print(f"\nSplit verification:")
print(f"Train:    {len(train):5} ({len(train)/total*100:.1f}%)")
print(f"Validate: {len(validate):5} ({len(validate)/total*100:.1f}%)")
print(f"Test:     {len(test):5} ({len(test)/total*100:.1f}%)")


train.to_csv("./data/train_subset.csv", index = False)
test.to_csv('./data/test_subset.csv', index = False)
validate.to_csv('./data/val_subset.csv', index = False)

print("\nData preparation complete!")
print("Files saved:")
print("  - data/train_subset.csv")
print("  - data/val_subset.csv")
print("  - data/test_subset.csv")