"""Model training, pruning, evaluation, and persistence."""
import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def train_baseline(X_train, y_train, random_state: int = 42):
    """Train an unpruned baseline Decision Tree."""
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_pruned(X_train, y_train, random_state: int = 42,
                 max_depth: int = 6, min_samples_leaf: int = 5):
    """Train a pruned Decision Tree."""
    model = DecisionTreeClassifier(
        random_state=random_state,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    """Return metrics dict and predictions."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'y_pred': y_pred
    }


def plot_decision_tree(model, feature_names, class_names, filepath: str,
                       max_depth_display: int = 4):
    """Save a decision tree visualization."""
    plt.figure(figsize=(24, 16))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        max_depth=max_depth_display,
        fontsize=8
    )
    plt.title('Decision Tree Visualization', fontsize=16)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    return filepath


def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from app.models.preprocess import prepare_data, build_preprocessor

    X_train, X_test, y_train, y_test, df = prepare_data('vgsales.csv')
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    cat_features = preprocessor.named_transformers_['cat'] \
        .named_steps['onehot'].get_feature_names_out(['Platform', 'Genre', 'Publisher'])
    num_features = ['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    feature_names = list(cat_features) + num_features
    class_names = ['Rendah', 'Sedang', 'Tinggi']

    baseline = train_baseline(X_train_t, y_train)
    base_metrics = evaluate(baseline, X_test_t, y_test)
    print("Baseline Decision Tree:")
    print(f"  Accuracy: {base_metrics['accuracy']}")
    print(f"  F1: {base_metrics['f1']}")

    pruned = train_pruned(X_train_t, y_train)
    pruned_metrics = evaluate(pruned, X_test_t, y_test)
    print("\nPruned Decision Tree:")
    print(f"  Accuracy: {pruned_metrics['accuracy']}")
    print(f"  F1: {pruned_metrics['f1']}")

    save_model(pruned, 'data/models/decision_tree_pruned.pkl')
    print("\nPruned model saved to data/models/decision_tree_pruned.pkl")

    os.makedirs('data/models', exist_ok=True)
    plot_decision_tree(
        pruned, feature_names, class_names,
        'data/models/decision_tree_viz.png', max_depth_display=4
    )
    print("Tree visualization saved to data/models/decision_tree_viz.png")

    metrics = {
        'baseline': base_metrics,
        'pruned': pruned_metrics
    }
    import json
    with open('data/models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print("Metrics saved to data/models/metrics.json")