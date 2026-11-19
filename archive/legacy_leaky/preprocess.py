"""Preprocessing pipeline for Game Sales Prediction."""
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
import joblib


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create Sales_Category target from Global_Sales."""
    df = df.copy()
    df['Sales_Category'] = pd.cut(
        df['Global_Sales'],
        bins=[-np.inf, 0.5, 2.0, np.inf],
        labels=['Rendah', 'Sedang', 'Tinggi']
    )
    return df


def get_feature_columns():
    """Return feature column names used for training."""
    return ['Platform', 'Genre', 'Publisher', 'Year',
            'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']


def build_preprocessor():
    """Build a ColumnTransformer for categorical and numeric features."""
    categorical_features = ['Platform', 'Genre', 'Publisher']
    numeric_features = ['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numeric_transformer, numeric_features)
        ]
    )
    return preprocessor


def prepare_data(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """Load data, create target, split into train/test."""
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path)
    df = create_target(df)

    X = df[get_feature_columns()]
    y = df['Sales_Category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, df


def save_preprocessor(preprocessor, path: str):
    joblib.dump(preprocessor, path)


def load_preprocessor(path: str):
    return joblib.load(path)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, df = prepare_data('vgsales.csv')
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Target distribution (train):\n{y_train.value_counts()}")
    print(f"Target distribution (test):\n{y_test.value_counts()}")

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    print(f"Transformed train shape: {X_train_t.shape}")

    cat_features = (preprocessor.named_transformers_['cat']
                    .named_steps['onehot'].get_feature_names_out(['Platform', 'Genre', 'Publisher']))
    num_features = ['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    all_features = list(cat_features) + num_features
    print(f"Total features: {len(all_features)}")

    save_preprocessor(preprocessor, 'data/models/preprocessor.pkl')
    print("Preprocessor saved to data/models/preprocessor.pkl")