import pandas as pd

# Load dataset (no header)
df = pd.read_csv("wdbc.data", header=None)
df.drop(columns=[0], inplace=True)
# Column 1 is the diagnosis label
diagnosis = ['diagnosis']

# Each feature group
features = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness',
            'concavity', 'concave_points', 'symmetry', 'fractal_dimension']

# Create full column name list
columns = diagnosis + [f'{feat}_{stat}' for stat in ['mean', 'se', 'worst'] for feat in features]

df.columns = columns
# 'M' = 1 (malignant), 'B' = 0 (benign)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
print(df.isnull().sum().sum())
df.duplicated().sum()
from sklearn.preprocessing import StandardScaler

X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
