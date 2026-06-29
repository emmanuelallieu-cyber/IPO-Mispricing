import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import shap
import re
import warnings

warnings.filterwarnings('ignore')

print("Loading Data...")
# 1. Load Data
X = pd.read_csv('X_Features_Engineered.csv')
y = pd.read_csv('y_Target.csv')

# Clean column names for LightGBM & XGBoost compatibility
X.columns = [re.sub(r'[\[\]<>,]+', '', col) for col in X.columns]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.2, random_state=42)

# 2. Initialize Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1)
}

results = []

# --- PLOT 1: ROC CURVE COMPARISON ---
print("Training models and calculating metrics...")
plt.figure(figsize=(10, 8))

for name, model in models.items():
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    results.append({'Model': name, 'Accuracy': acc, 'AUC-ROC': roc_auc})
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)

# SAVE INSTEAD OF SHOW
plt.savefig('ROC_Curve_Comparison.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: ROC_Curve_Comparison.png")

print("\n--- Model Performance ---")
print(pd.DataFrame(results).to_string(index=False))
print("\n")

# --- PLOT 2: INDIVIDUAL FEATURE IMPORTANCES ---
print("Generating Feature Importance Charts...")
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
tree_models = ['Random Forest', 'XGBoost', 'LightGBM']

for i, name in enumerate(tree_models):
    model = models[name]
    importances = model.feature_importances_
    
    # Get top 15 features
    indices = np.argsort(importances)[-15:]
    
    axes[i].barh(range(len(indices)), importances[indices], align='center', color='teal')
    axes[i].set_yticks(range(len(indices)))
    axes[i].set_yticklabels([X.columns[j] for j in indices])
    axes[i].set_xlabel('Relative Importance')
    axes[i].set_title(f'Top 15 Features: {name}')
    axes[i].grid(axis='x', alpha=0.3)

plt.tight_layout()

# SAVE INSTEAD OF SHOW
plt.savefig('Feature_Importances.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: Feature_Importances.png")

# --- PLOT 3: EXPLAINABLE AI (SHAP) ---
print("Generating XAI SHAP Summary Plot for LightGBM...")
lgb_model = models['LightGBM']

# Initialize the SHAP Tree Explainer
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_test)

# For LightGBM binary classification, shap_values might be a list. We take the values for class 1.
if isinstance(shap_values, list):
    shap_values_to_plot = shap_values[1]
else:
    shap_values_to_plot = shap_values

# Plot the SHAP summary
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_to_plot, X_test, max_display=15, show=False)
plt.title('SHAP Summary Plot (LightGBM) - Impact on IPO Success')
plt.tight_layout()

# SAVE INSTEAD OF SHOW
plt.savefig('SHAP_Summary_LightGBM.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: SHAP_Summary_LightGBM.png")

print("\nExecution Complete! All files have been saved to your current directory.")