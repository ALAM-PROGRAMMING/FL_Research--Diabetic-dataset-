import pandas as pd
from sklearn.model_selection import train_test_split

india_df = pd.read_csv("../data/processed/india_cleaned_aligned.csv")
us_df = pd.read_csv("../data/processed/us_cleaned_aligned.csv")

centralized_df = pd.concat([india_df, us_df], axis=0).reset_index(drop=True)

X = centralized_df.drop(columns=["Diabetes_binary"])
y = centralized_df["Diabetes_binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
y_proba_lr = lr_model.predict_proba(X_test)[:, 1]
from sklearn.metrics import accuracy_score, classification_report

lr_accuracy = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy:", lr_accuracy)

print(classification_report(y_test, y_pred_lr))
from sklearn.metrics import confusion_matrix

cm_lr = confusion_matrix(y_test, y_pred_lr)
print(cm_lr)
from sklearn.metrics import roc_auc_score

lr_auc = roc_auc_score(y_test, y_proba_lr)
print("ROC-AUC Score:", lr_auc)
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import os

save_path = "../4_evaluation_and_paper/"
os.makedirs(save_path, exist_ok=True)

fpr, tpr, _ = roc_curve(y_test, y_proba_lr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {lr_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()

plt.savefig(save_path + "logistic_regression_roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

threshold = 0.45
y_pred_rf = (y_proba_rf >= threshold).astype(int)
y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
from sklearn.metrics import accuracy_score, classification_report

rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", rf_accuracy)

print(classification_report(y_test, y_pred_rf))
from sklearn.metrics import confusion_matrix

cm_rf = confusion_matrix(y_test, y_pred_rf)
print(cm_rf)
from sklearn.metrics import roc_auc_score

rf_auc = roc_auc_score(y_test, y_proba_rf)
print("Random Forest ROC-AUC:", rf_auc)
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import os

save_path = "../4_evaluation_and_paper/"
os.makedirs(save_path, exist_ok=True)

fpr, tpr, _ = roc_curve(y_test, y_proba_rf)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {rf_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()

plt.savefig(save_path + "random_forest_roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
from xgboost import XGBClassifier
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
    use_label_encoder=False
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
from sklearn.metrics import confusion_matrix

cm_xgb = confusion_matrix(y_test, y_pred_xgb)
print(cm_xgb)
from sklearn.metrics import roc_auc_score

xgb_auc = roc_auc_score(y_test, y_proba_xgb)
print("XGBoost ROC-AUC:", xgb_auc)
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import os

save_path = "../4_evaluation_and_paper/"
os.makedirs(save_path, exist_ok=True)

fpr, tpr, _ = roc_curve(y_test, y_proba_xgb)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {xgb_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost")
plt.legend()

plt.savefig(save_path + "xgboost_roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()
from sklearn.ensemble import VotingClassifier
voting_model = VotingClassifier(
    estimators=[
        ("lr", lr_model),
        ("rf", rf_model),
        ("xgb", xgb_model)
    ],
    voting="soft",
    weights=[1, 1, 2],   # Give XGBoost more weight
    n_jobs=-1
)
voting_model.fit(X_train, y_train)
y_proba_vote = voting_model.predict_proba(X_test)[:, 1]
y_pred_vote = (y_proba_vote >= 0.5).astype(int)
from sklearn.metrics import accuracy_score, classification_report

vote_accuracy = accuracy_score(y_test, y_pred_vote)
print("Voting Accuracy:", vote_accuracy)

print(classification_report(y_test, y_pred_vote))
from sklearn.metrics import confusion_matrix

cm_vote = confusion_matrix(y_test, y_pred_vote)
print("Confusion Matrix:\n", cm_vote)
from sklearn.metrics import roc_auc_score

vote_auc = roc_auc_score(y_test, y_proba_vote)
print("Voting ROC-AUC:", vote_auc)
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import os

save_path = "../4_evaluation_and_paper/"
os.makedirs(save_path, exist_ok=True)

fpr, tpr, _ = roc_curve(y_test, y_proba_vote)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {vote_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Voting Classifier")
plt.legend()

plt.savefig(save_path + "voting_classifier_roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

accuracy_list = []
auc_list = []
recall_list = []
f1_list = []

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

for train_idx, test_idx in sss.split(X, y):
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=None
    )
    
    model.fit(X_train, y_train)
    
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    
    
accuracy_list.append(accuracy_score(y_test, y_pred))
auc_list.append(roc_auc_score(y_test, y_proba))
recall_list.append(recall_score(y_test, y_pred))
f1_list.append(f1_score(y_test, y_pred))

print("Accuracy: {:.3f} ± {:.3f}".format(np.mean(accuracy_list), np.std(accuracy_list)))
print("ROC-AUC: {:.3f} ± {:.3f}".format(np.mean(auc_list), np.std(auc_list)))
print("Recall: {:.3f} ± {:.3f}".format(np.mean(recall_list), np.std(recall_list)))
print("F1-score: {:.3f} ± {:.3f}".format(np.mean(f1_list), np.std(f1_list)))
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import os

# 1. Create the plots folder
os.makedirs("../4_evaluation_and_paper/plots/", exist_ok=True)

# 2. THE FIX: Force recalculate the probabilities to clear notebook memory bugs
# We specifically slice [:, 1] to get the probability of Class 1 (Diabetes)
fresh_proba_lr = lr_model.predict_proba(X_test)[:, 1]
fresh_proba_rf = rf_model.predict_proba(X_test)[:, 1]
fresh_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
fresh_proba_vote = voting_model.predict_proba(X_test)[:, 1]

# 3. Calculate ROC curves using the fresh probabilities
fpr_lr, tpr_lr, _ = roc_curve(y_test, fresh_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, fresh_proba_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, fresh_proba_xgb)
fpr_vote, tpr_vote, _ = roc_curve(y_test, fresh_proba_vote)

# --- The rest of the plotting code remains exactly the same ---
plt.figure(figsize=(8, 6))
plt.style.use('default') 

plt.plot(fpr_lr, tpr_lr, color='#1f77b4', linestyle=':', linewidth=2, label=f'Logistic Regression (AUC = {lr_auc:.3f})')
plt.plot(fpr_rf, tpr_rf, color='#2ca02c', linestyle='-.', linewidth=2, label=f'Random Forest (AUC = {rf_auc:.3f})')
plt.plot(fpr_xgb, tpr_xgb, color='#ff7f0e', linestyle='--', linewidth=2.5, label=f'XGBoost (AUC = {xgb_auc:.3f})')
plt.plot(fpr_vote, tpr_vote, color='#d62728', linestyle='-', linewidth=3, label=f'Soft Voting Ensemble (AUC = {vote_auc:.3f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Random Guessing (AUC = 0.500)')

plt.title('Receiver Operating Characteristic (ROC) - Centralized Baselines', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.legend(loc='lower right', fontsize=10, frameon=True, shadow=True)
plt.grid(True, linestyle='--', alpha=0.6)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

save_path_pdf = "../4_evaluation_and_paper/plots/unified_roc_curve.pdf"
save_path_png = "../4_evaluation_and_paper/plots/unified_roc_curve.png"

plt.savefig(save_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
plt.savefig(save_path_png, format='png', dpi=300, bbox_inches='tight')

plt.show()