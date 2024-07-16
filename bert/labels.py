import pandas as pd

path = 'data.csv'

df = pd.read_csv(path)

value_counts = df['target'].value_counts()
df['labels'] = df['target'].apply(lambda x: x if value_counts[x] >= 5 else 'Другое')

labels = df['labels'].unique()
