"""
=============================================================================
Smart Healthcare Prediction System — Diabetes Risk Assessment
=============================================================================
Dataset : UCI Diabetes 130-US Hospitals (1999–2008), ID=296
          Strack et al., BioMed Research International 2014
Approach: Ensemble Learning — RF, GBM, AdaBoost, Voting Classifier
Target   : 0=Non-Diabetic | 1=Pre-Diabetic | 2=Diabetic
=============================================================================
"""

# ── Standard imports ────────────────────────────────────────────────────────
import os, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, VotingClassifier)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              ConfusionMatrixDisplay)
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE, 'data', 'diabetes_raw.csv')
PLOTS_DIR   = os.path.join(BASE, 'plots')
MODELS_DIR  = os.path.join(BASE, 'models')
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

CLASS_NAMES = ['Non-Diabetic', 'Pre-Diabetic', 'Diabetic']


# =============================================================================
# SECTION 1 — DATASET LOADING & STRUCTURE
# =============================================================================

def load_dataset():
    """Load dataset and display structural overview."""
    print("\n" + "="*70)
    print("SECTION 1: DATASET LOADING & STRUCTURE")
    print("="*70)

    df = pd.read_csv(DATA_PATH)

    print(f"\n  Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory usage   : {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print(f"\n  Target distribution (diabetes_class):")
    vc = df['diabetes_class'].value_counts().sort_index()
    for cls, cnt in vc.items():
        print(f"    Class {cls} ({CLASS_NAMES[cls]}): {cnt:,}  ({cnt/len(df)*100:.1f}%)")

    # Identify column types
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    num_cols = [c for c in num_cols if c != 'diabetes_class']

    print(f"\n  Categorical columns ({len(cat_cols)}): {cat_cols[:5]}...")
    print(f"  Numerical columns  ({len(num_cols)}): {num_cols[:5]}...")

    print(f"\n  First 3 rows preview:")
    print(df.head(3).to_string())
    print(f"\n  Statistical summary (numerical):")
    print(df[num_cols[:8]].describe().round(2).to_string())

    return df, cat_cols, num_cols


# =============================================================================
# SECTION 2 — EXPLORATORY DATA ANALYSIS
# =============================================================================

def run_eda(df, cat_cols, num_cols):
    """Comprehensive EDA with visualizations and textual insights."""
    print("\n" + "="*70)
    print("SECTION 2: EXPLORATORY DATA ANALYSIS")
    print("="*70)

    # ── 2a. Missing values ───────────────────────────────────────────────────
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)

    if len(missing_df) > 0:
        print(f"\n  Columns with missing values:\n{missing_df.head(10).to_string()}")
    else:
        print("\n  No missing values in numerical columns.")

    # Missing value heatmap
    plt.figure(figsize=(14, 4))
    sns.heatmap(df[num_cols].isnull().T, cbar=False, yticklabels=True, cmap='YlOrRd')
    plt.title('Missing Values Heatmap (Numerical Features)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'eda_missing_values.png'), dpi=120)
    plt.close()

    # ── 2b. Class distribution ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    vc = df['diabetes_class'].value_counts().sort_index()
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    axes[0].bar([CLASS_NAMES[i] for i in vc.index], vc.values, color=colors, edgecolor='black', alpha=0.85)
    axes[0].set_title('Target Class Distribution', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(vc.values):
        axes[0].text(i, v + 30, f'{v:,}\n({v/len(df)*100:.1f}%)', ha='center', fontsize=9)
    axes[1].pie(vc.values, labels=[CLASS_NAMES[i] for i in vc.index],
                autopct='%1.1f%%', colors=colors, startangle=90, wedgeprops={'edgecolor':'white','linewidth':1.5})
    axes[1].set_title('Class Distribution (Pie)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'eda_class_distribution.png'), dpi=120)
    plt.close()

    # ── 2c. Numerical feature distributions ─────────────────────────────────
    key_num = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
               'num_medications', 'number_diagnoses', 'number_inpatient',
               'glucose_level', 'hba1c_level']
    key_num = [c for c in key_num if c in df.columns]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(key_num):
        for cls, color, label in zip([0,1,2], colors, CLASS_NAMES):
            subset = df[df['diabetes_class'] == cls][col]
            axes[i].hist(subset, bins=25, alpha=0.55, color=color, label=label, density=True)
        axes[i].set_title(col.replace('_',' ').title(), fontsize=10, fontweight='bold')
        axes[i].set_xlabel('Value'); axes[i].set_ylabel('Density')
        axes[i].legend(fontsize=7)
    plt.suptitle('Feature Distributions by Diabetes Class', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'eda_distributions.png'), dpi=120, bbox_inches='tight')
    plt.close()

    # ── 2d. Correlation heatmap ──────────────────────────────────────────────
    corr_cols = num_cols + ['diabetes_class']
    corr_cols = [c for c in corr_cols if c in df.columns]
    corr = df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(14, 11))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, linewidths=0.4, annot_kws={'size': 7},
                cbar_kws={'shrink': 0.8})
    plt.title('Correlation Heatmap — Numerical Features', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'eda_correlation_heatmap.png'), dpi=120)
    plt.close()

    # ── 2e. Outlier detection (IQR boxplots) ────────────────────────────────
    plot_cols = ['time_in_hospital', 'num_lab_procedures', 'num_medications',
                 'number_inpatient', 'glucose_level', 'hba1c_level']
    plot_cols = [c for c in plot_cols if c in df.columns]
    fig, axes = plt.subplots(1, len(plot_cols), figsize=(16, 5))
    for i, col in enumerate(plot_cols):
        data_by_class = [df[df['diabetes_class']==cls][col].dropna() for cls in [0,1,2]]
        bp = axes[i].boxplot(data_by_class, patch_artist=True, notch=False,
                              medianprops={'color':'black','linewidth':2})
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[i].set_title(col.replace('_',' ').title(), fontsize=9, fontweight='bold')
        axes[i].set_xticklabels(['Non-D.','Pre-D.','Diab.'], fontsize=8)
    plt.suptitle('Outlier Detection via Boxplots (by Class)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'eda_outliers_boxplot.png'), dpi=120)
    plt.close()

    # ── 2f. Key EDA Insights (printed) ──────────────────────────────────────
    print("\n  KEY EDA INSIGHTS:")
    print("  " + "-"*60)
    for cls in [0,1,2]:
        sub = df[df['diabetes_class'] == cls]
        print(f"\n  Class {cls} — {CLASS_NAMES[cls]} (n={len(sub):,})")
        for col in ['num_medications','time_in_hospital','number_inpatient']:
            if col in df.columns:
                print(f"    {col:30s}: mean={sub[col].mean():.2f}, median={sub[col].median():.1f}")
        if 'hba1c_level' in df.columns:
            print(f"    {'hba1c_level':30s}: mean={sub['hba1c_level'].mean():.2f}")
        if 'glucose_level' in df.columns:
            print(f"    {'glucose_level':30s}: mean={sub['glucose_level'].mean():.2f}")

    # Class imbalance note
    vc = df['diabetes_class'].value_counts()
    imbalance_ratio = vc.max() / vc.min()
    print(f"\n  Class imbalance ratio (max/min): {imbalance_ratio:.1f}x  → SMOTE required")

    # Correlation with target
    corr_target = df[corr_cols].corr()['diabetes_class'].drop('diabetes_class').abs().sort_values(ascending=False)
    print(f"\n  Top 5 features correlated with target:")
    for feat, val in corr_target.head(5).items():
        print(f"    {feat:30s}: |r| = {val:.4f}")

    print("\n  ✓ EDA plots saved to plots/")
    return corr_target


# =============================================================================
# SECTION 3 — DATA PREPROCESSING
# =============================================================================

def preprocess(df, cat_cols, num_cols):
    """Full preprocessing pipeline: imputation, encoding, scaling, SMOTE."""
    print("\n" + "="*70)
    print("SECTION 3: DATA PREPROCESSING")
    print("="*70)

    df = df.copy()

    # ── 3a. Drop near-zero-variance / redundant columns ─────────────────────
    # Many medication columns are >99% 'No' — collapse to binary
    cols_to_drop = []
    for col in cat_cols:
        if col in df.columns:
            top_val_pct = df[col].value_counts(normalize=True).iloc[0]
            if top_val_pct > 0.98:  # >98% same value → low information
                cols_to_drop.append(col)
    print(f"\n  Dropping {len(cols_to_drop)} near-zero-variance categorical columns.")
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Update cat_cols
    cat_cols = [c for c in cat_cols if c not in cols_to_drop and c in df.columns]

    # ── 3b. Handle '?' as missing in categorical ─────────────────────────────
    for col in cat_cols:
        df[col] = df[col].replace('?', np.nan)

    # ── 3c. Imputation ───────────────────────────────────────────────────────
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    num_feat = [c for c in num_cols if c in df.columns and c != 'diabetes_class']
    df[num_feat] = num_imputer.fit_transform(df[num_feat])

    for col in cat_cols:
        df[col] = cat_imputer.fit_transform(df[[col]]).ravel()

    print(f"  Imputation complete. Missing after: {df.isnull().sum().sum()}")

    # ── 3d. Label Encoding for categorical features ──────────────────────────
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    print(f"  Label-encoded {len(cat_cols)} categorical columns.")

    # ── 3e. Feature/target split ─────────────────────────────────────────────
    X = df.drop(columns=['diabetes_class'])
    y = df['diabetes_class']
    feature_names = list(X.columns)
    print(f"\n  Features: {X.shape[1]}, Samples: {X.shape[0]}")

    # ── 3f. Train/test split (stratified, 80/20) ─────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"  Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    # ── 3g. StandardScaler ───────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── 3h. SMOTE on training set only ──────────────────────────────────────
    print(f"\n  Before SMOTE: {dict(pd.Series(y_train).value_counts().sort_index())}")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print(f"  After  SMOTE: {dict(pd.Series(y_train_res).value_counts().sort_index())}")

    # Save artifacts
    joblib.dump(scaler,   os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(encoders, os.path.join(MODELS_DIR, 'encoders.pkl'))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, 'feature_names.pkl'))
    joblib.dump(cat_cols,  os.path.join(MODELS_DIR, 'cat_cols.pkl'))
    joblib.dump(num_feat,  os.path.join(MODELS_DIR, 'num_feat.pkl'))

    print("\n  ✓ Preprocessing complete. Artifacts saved.")
    return (X_train_res, X_test_scaled, y_train_res, y_test,
            feature_names, scaler, encoders)


# =============================================================================
# SECTION 4 — MODEL DEVELOPMENT
# =============================================================================

def train_models(X_train, X_test, y_train, y_test, feature_names):
    """Train RF, GBM, AdaBoost, Voting Classifier with cross-validation."""
    print("\n" + "="*70)
    print("SECTION 4: MODEL DEVELOPMENT")
    print("="*70)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Model definitions ────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_split=5,
        min_samples_leaf=2, max_features='sqrt',
        class_weight='balanced', n_jobs=-1, random_state=42)

    gbm = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.08, max_depth=5,
        min_samples_split=10, subsample=0.85, random_state=42)

    ada = AdaBoostClassifier(
        n_estimators=150, learning_rate=0.1, random_state=42)

    voting = VotingClassifier(
        estimators=[('rf', rf), ('gbm', gbm), ('ada', ada)],
        voting='soft', n_jobs=-1)

    models = {
        'Random Forest':         rf,
        'Gradient Boosting':     gbm,
        'AdaBoost':              ada,
        'Voting Classifier':     voting
    }

    trained = {}
    cv_results = {}

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        trained[name] = model

        # 5-fold CV on training data
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=cv, scoring='f1_weighted', n_jobs=-1)
        cv_results[name] = cv_scores
        print(f"    CV F1 (weighted): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Save model
        safe_name = name.lower().replace(' ', '_')
        joblib.dump(model, os.path.join(MODELS_DIR, f'{safe_name}.pkl'))

    # Feature importance from RF
    if hasattr(rf, 'feature_importances_'):
        fi = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        fi.head(20).plot(kind='barh', color='steelblue', edgecolor='black', alpha=0.8)
        plt.gca().invert_yaxis()
        plt.xlabel('Feature Importance (Gini)', fontsize=11)
        plt.title('Random Forest — Top 20 Feature Importances', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance_rf.png'), dpi=120)
        plt.close()
        print(f"\n  Top 5 RF features: {fi.head(5).index.tolist()}")

    return trained, cv_results


# =============================================================================
# SECTION 5 — MODEL EVALUATION
# =============================================================================

def evaluate_models(trained, X_test, y_test):
    """Full evaluation: metrics per class + confusion matrices + comparison."""
    print("\n" + "="*70)
    print("SECTION 5: MODEL EVALUATION")
    print("="*70)

    results = {}
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for idx, (name, model) in enumerate(trained.items()):
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)

        results[name] = {'Accuracy': acc, 'Precision': prec,
                         'Recall': rec, 'F1-Score': f1}

        print(f"\n  {name}")
        print(f"    Accuracy : {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall   : {rec:.4f}")
        print(f"    F1-Score : {f1:.4f}")
        print(f"\n    Classification Report:")
        print(classification_report(y_test, y_pred, target_names=CLASS_NAMES,
                                    zero_division=0))

        # Confusion matrix subplot
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=CLASS_NAMES)
        disp.plot(ax=axes[idx], colorbar=False, cmap='Blues')
        axes[idx].set_title(f'{name}\nAcc={acc:.3f} | F1={f1:.3f}',
                             fontsize=11, fontweight='bold')
        axes[idx].set_xticklabels(CLASS_NAMES, rotation=15, ha='right', fontsize=9)
        axes[idx].set_yticklabels(CLASS_NAMES, fontsize=9)

    plt.suptitle('Confusion Matrices — All Models', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrices.png'), dpi=120)
    plt.close()

    return results


# =============================================================================
# SECTION 6 — MODEL COMPARISON
# =============================================================================

def compare_models(results, cv_results):
    """Comparison table + bar chart + best model analysis."""
    print("\n" + "="*70)
    print("SECTION 6: MODEL COMPARISON")
    print("="*70)

    df_results = pd.DataFrame(results).T.round(4)
    df_results['CV F1 Mean'] = [cv_results[m].mean().round(4) for m in df_results.index]
    df_results['CV F1 Std']  = [cv_results[m].std().round(4)  for m in df_results.index]

    print("\n  COMPARISON TABLE:")
    print("  " + "-"*70)
    print(df_results.to_string())
    print("  " + "-"*70)

    best_model = df_results['F1-Score'].idxmax()
    print(f"\n  Best model: {best_model}  (F1={df_results.loc[best_model,'F1-Score']:.4f})")

    # Bar comparison chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics))
    model_names = list(results.keys())
    width = 0.2
    colors_m = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (name, color) in enumerate(zip(model_names, colors_m)):
        vals = [results[name][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=name, color=color,
                      alpha=0.85, edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, rotation=0)

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison.png'), dpi=120)
    plt.close()

    # CV score distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    cv_data = [cv_results[m] for m in model_names]
    bp = ax.boxplot(cv_data, patch_artist=True, notch=False,
                    medianprops={'color':'black','linewidth':2})
    for patch, color in zip(bp['boxes'], colors_m):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.set_xticklabels(model_names, rotation=10, fontsize=10)
    ax.set_ylabel('F1-Score (Weighted)', fontsize=11)
    ax.set_title('5-Fold Cross-Validation Score Distribution', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'cv_comparison.png'), dpi=120)
    plt.close()

    print("\n  ✓ Comparison plots saved.")
    return best_model, df_results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("\n" + "★"*70)
    print("  SMART HEALTHCARE PREDICTION SYSTEM — DIABETES RISK ASSESSMENT")
    print("  Dataset: UCI Diabetes 130-US Hospitals (Strack et al. 2014)")
    print("★"*70)

    # Run pipeline
    df, cat_cols, num_cols = load_dataset()
    corr_target = run_eda(df, cat_cols, num_cols)
    X_train, X_test, y_train, y_test, feat_names, scaler, encoders = preprocess(df, cat_cols, num_cols)
    trained, cv_results = train_models(X_train, X_test, y_train, y_test, feat_names)
    results = evaluate_models(trained, X_test, y_test)
    best_model, comparison_df = compare_models(results, cv_results)

    print("\n" + "="*70)
    print("  PIPELINE COMPLETE")
    print(f"  Best Model  : {best_model}")
    print(f"  Models saved: {MODELS_DIR}/")
    print(f"  Plots saved : {PLOTS_DIR}/")
    print("="*70 + "\n")
