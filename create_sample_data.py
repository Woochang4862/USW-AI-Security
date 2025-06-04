import pandas as pd
import numpy as np
from pathlib import Path

# Load and sample data
df = pd.read_csv('MMTD/DATA/email_data/EDP.csv')
print(f'Original dataset: {len(df)} samples')

# Sample 1000 samples for quick test (500 from each class)
spam_samples = df[df['labels'] == 1].sample(n=500, random_state=42)
ham_samples = df[df['labels'] == 0].sample(n=500, random_state=42)
sample_df = pd.concat([spam_samples, ham_samples]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f'Sample dataset: {len(sample_df)} samples')
print('Label distribution:', sample_df['labels'].value_counts().to_dict())

# Check if images exist
data_path = Path('MMTD/DATA/email_data/pics')
valid_count = 0
missing_count = 0
for idx, row in sample_df.head(20).iterrows():
    image_path = data_path / row['pics']
    if image_path.exists():
        valid_count += 1
    else:
        missing_count += 1
        if missing_count <= 5:  # Only print first 5 missing
            print(f'Missing: {image_path}')

print(f'Valid images in first 20: {valid_count}/20')

# Save sample for testing
sample_df.to_csv('MMTD/DATA/email_data/EDP_sample.csv', index=False)
print('Sample saved to EDP_sample.csv') 