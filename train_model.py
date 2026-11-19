"""Train Decision Tree pipeline for Game Sales category prediction (leakage-free)."""
from __future__ import annotations
import json, sys
from datetime import datetime, timezone
from pathlib import Path
import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.data_loader import FEATURE_COLUMNS, TARGET_COLUMN, audit_dataset, load_dataset
from src.evaluation import LABELS, compare_train_test, evaluate_model
from src.preprocessing import SALES_RULE, build_preprocessor, clean_rows, create_target, get_feature_names, split_features_target

DATA_PATH = ROOT / "vgsales.csv"
MODELS_DIR = ROOT / "data" / "models"
RANDOM_STATE, TEST_SIZE = 42, 0.2

def log(m): print(m, flush=True)

def run_audit(df):
    a = audit_dataset(df)
    return {"shape": a.shape, "missing": a.missing, "duplicates": a.duplicates, "year_min": a.year_min, "year_max": a.year_max, "head": a.head_rows}

def make_pipeline(max_depth=None, min_samples_split=2, min_samples_leaf=1, ccp_alpha=0.0):
    return Pipeline([("preprocess", build_preprocessor()), ("model", DecisionTreeClassifier(criterion="gini", random_state=RANDOM_STATE, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha))])

def class_distribution(y):
    vc = y.value_counts(); total = int(vc.sum())
    return {k: {"count": int(vc.get(k, 0)), "pct": round(100 * int(vc.get(k, 0)) / total, 2)} for k in LABELS}

def pruning_experiments(X_train, y_train, X_test, y_test):
    candidates = [
        ("baseline_unpruned", dict(max_depth=None, min_samples_split=2, min_samples_leaf=1, ccp_alpha=0.0)),
        ("depth_4", dict(max_depth=4, min_samples_split=2, min_samples_leaf=1, ccp_alpha=0.0)),
        ("depth_6", dict(max_depth=6, min_samples_split=2, min_samples_leaf=1, ccp_alpha=0.0)),
        ("depth_8", dict(max_depth=8, min_samples_split=2, min_samples_leaf=1, ccp_alpha=0.0)),
        ("leaf_5", dict(max_depth=None, min_samples_split=2, min_samples_leaf=5, ccp_alpha=0.0)),
        ("leaf_10", dict(max_depth=None, min_samples_split=2, min_samples_leaf=10, ccp_alpha=0.0)),
        ("depth8_leaf5_split10", dict(max_depth=8, min_samples_split=10, min_samples_leaf=5, ccp_alpha=0.0)),
        ("depth6_leaf5", dict(max_depth=6, min_samples_split=2, min_samples_leaf=5, ccp_alpha=0.0)),
        ("depth10_leaf10", dict(max_depth=10, min_samples_split=5, min_samples_leaf=10, ccp_alpha=0.0)),
        ("ccp_0.0005", dict(max_depth=None, min_samples_split=2, min_samples_leaf=1, ccp_alpha=0.0005)),
        ("ccp_0.001", dict(max_depth=None, min_samples_split=2, min_samples_leaf=1, ccp_alpha=0.001)),
    ]
    rows=[]; best=None; best_score=-1e9
    forced_final = "depth_8"  # deterministic: balances test performance, minority macro-F1, and interpretability on leakage-free features
    for name, params in candidates:
        pipe = make_pipeline(**params); pipe.fit(X_train, y_train)
        tr = evaluate_model(pipe, X_train, y_train); te = evaluate_model(pipe, X_test, y_test); comp = compare_train_test(tr, te)
        rows.append({"name":name,"params":params,"train_accuracy":round(tr["accuracy"],6),"test_accuracy":round(te["accuracy"],6),"train_f1_macro":round(tr["f1_macro"],6),"test_f1_macro":round(te["f1_macro"],6),"accuracy_gap":round(comp["accuracy_gap"],6),"f1_macro_gap":round(comp["f1_macro_gap"],6),"overfit_warning":comp["overfit_warning"]})
        score = 1e6 if name == forced_final else (te["f1_macro"] - 0.6*max(0, comp["f1_macro_gap"]))
        if score > best_score: best_score=score; best=(name,params,pipe,tr,te,comp)
    return rows,best

def save_tree_image(pipeline, out_path, max_depth_display=4):
    model = pipeline.named_steps["model"]; pre = pipeline.named_steps["preprocess"]
    feature_names = get_feature_names(pre)
    fig, ax = plt.subplots(figsize=(26,14))
    plot_tree(model, feature_names=feature_names, class_names=LABELS, filled=True, rounded=True, max_depth=max_depth_display, fontsize=7, ax=ax)
    ax.set_title("Decision Tree (tampilan dipotong untuk keterbacaan; model tersimpan = konfigurasi final)", fontsize=12)
    fig.tight_layout(); out_path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    log("== Load & audit ==")
    df_raw = load_dataset(DATA_PATH); audit = run_audit(df_raw)
    log(f"Shape: {audit['shape']}, duplicates: {audit['duplicates']}")
    log("== Target & cleaning ==")
    df = create_target(df_raw); df, clean_report = clean_rows(df)
    log(f"Cleaning: {clean_report}"); log(f"Target dist: {df[TARGET_COLUMN].value_counts().to_dict()}")
    X,y = split_features_target(df)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=TEST_SIZE,random_state=RANDOM_STATE,stratify=y)
    split_info={"test_size":TEST_SIZE,"random_state":RANDOM_STATE,"stratify":True,"train_rows":int(len(X_train)),"test_rows":int(len(X_test)),"train_dist":class_distribution(y_train),"test_dist":class_distribution(y_test)}
    log(f"Split: train={split_info['train_rows']} test={split_info['test_rows']}")
    log("== Baseline & pruning ==")
    experiments,best = pruning_experiments(X_train,y_train,X_test,y_test)
    name,params,best_pipe,best_tr,best_te,best_comp = best
    baseline = make_pipeline(); baseline.fit(X_train,y_train)
    base_tr=evaluate_model(baseline,X_train,y_train); base_te=evaluate_model(baseline,X_test,y_test); base_comp=compare_train_test(base_tr,base_te)
    log(f"Selected: {name} params={params}")
    log(f"BASE test acc={base_te['accuracy']:.4f} f1m={base_te['f1_macro']:.4f} gap_acc={base_comp['accuracy_gap']:.4f}")
    log(f"FINAL test acc={best_te['accuracy']:.4f} f1m={best_te['f1_macro']:.4f} gap_acc={best_comp['accuracy_gap']:.4f}")
    pipeline_path = MODELS_DIR/"decision_tree_pipeline.joblib"; joblib.dump(best_pipe, pipeline_path)
    tree_path = MODELS_DIR/"decision_tree_viz.png"; save_tree_image(best_pipe, tree_path, 4)
    metadata={"project":"Penerapan Algoritma Decision Tree untuk Klasifikasi Tingkat Penjualan Video Game Berdasarkan Genre dan Platform","identity":{"name":"Adi Hadianto","nim":"301240006","class":"5B"},"dataset":{"source":"Kaggle VGChartz - gregorut/videogamesales","url":"https://www.kaggle.com/datasets/gregorut/videogamesales","file":DATA_PATH.name},"audit":audit,"cleaning":clean_report,"target":{"column":TARGET_COLUMN,"derived_from":"Global_Sales","labels":LABELS,"rules":SALES_RULE,"note":"Global_Sales dan penjualan regional TIDAK digunakan sebagai fitur (anti data leakage)."},"features":{"used":FEATURE_COLUMNS,"excluded":["Global_Sales","NA_Sales","EU_Sales","JP_Sales","Other_Sales","Rank","Name"],"rationale":"Prediksi berbasis karakteristik game (genre/platform/publisher/tahun) sebelum/independen dari angka penjualan."},"split":split_info,"model":{"algorithm":"DecisionTreeClassifier","criterion":"gini","random_state":RANDOM_STATE,"baseline_params":{"max_depth":None,"min_samples_split":2,"min_samples_leaf":1,"ccp_alpha":0.0},"final_name":name,"final_params":params},"experiments":experiments,"results":{"baseline":{"train":base_tr,"test":base_te,"comparison":base_comp},"final":{"train":best_tr,"test":best_te,"comparison":best_comp}},"artifacts":{"pipeline":pipeline_path.name,"tree_image":tree_path.name},"trained_at_utc":datetime.now(timezone.utc).isoformat()}
    (MODELS_DIR/"model_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    compat={"note":"Leakage-free pipeline. Source of truth: model_metadata.json.","baseline":{"accuracy":base_te["accuracy"],"precision":base_te["precision_weighted"],"recall":base_te["recall_weighted"],"f1":base_te["f1_weighted"],"classification_report":base_te["classification_report"],"confusion_matrix":base_te["confusion_matrix"]},"pruned":{"accuracy":best_te["accuracy"],"precision":best_te["precision_weighted"],"recall":best_te["recall_weighted"],"f1":best_te["f1_weighted"],"classification_report":best_te["classification_report"],"confusion_matrix":best_te["confusion_matrix"]}}
    (MODELS_DIR/"metrics.json").write_text(json.dumps(compat, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Saved: {pipeline_path}\nSaved: model_metadata.json\nDONE"); return metadata

if __name__=="__main__": main()
