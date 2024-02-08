import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from scipy.io import arff

# Replace 'your_text_file.txt' with the actual path to your text file
text_file_path = 'bone-marrow.arff.txt'

# Load the text file into a DataFrame
df = pd.read_csv(text_file_path, sep='\t')

# Convert all columns to numeric, coerce errors to null values
for c in df.columns:
    #df[c] = pd.to_numeric(df[c], errors='coerc')

# Make sure binary columns are encoded as 0 and 1
for c in df.columns[df.nunique() == 2]:
    df[c] = (df[c] == 1) * 1.0

# Set target variable and features
y = df.survival_status
X = df.drop(columns=['survival_time', 'survival_status'])

# Split data into train/test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

# Define numeric and categorical columns
num_cols = X.columns[X.nunique() > 7]
cat_cols = X.columns[X.nunique() <= 7]

# Create preprocessing pipelines
cat_pipe = Pipeline([("imputer", SimpleImputer(strategy='most_frequent')),
                    ("ohe", OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore'))])

num_pipe = Pipeline([("imputer", SimpleImputer(strategy='mean')),
                    ("scale", StandardScaler())])

# Create column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", cat_pipe, cat_cols),
        ("num", num_pipe, num_cols)
    ]
)

# Create the main pipeline with preprocessing, PCA, and random forest classifier
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("pca", PCA()),
    ("clf", RandomForestClassifier(random_state=42))
])

# Fit the pipeline on the training data
pipeline.fit(x_train, y_train)

# Convert the DataFrame to ARFF format
arff_data = pd.DataFrame(pipeline.transform(X), columns=X.columns)
arff_data['survival_status'] = y  # Add the target variable back

# Save the ARFF file
arff_data.to_csv('your_file.arff', index=False)
