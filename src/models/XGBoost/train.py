import os
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, precision_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

def run_kfold_training():
    train_data = pd.read_csv("data/processed/train.csv")
    auc_scores = []
    f1_scores = []
    accuracy_scores = []
    recall_scores = []  
    precision_scores = []
    
    best_auc = 0
    best_model_auc = None
    best_f1 = 0
    best_model_f1 = None
    best_recall = 0
    best_model_recall = None
    best_accuracy = 0
    best_model_accuracy = None
    best_precision = 0
    best_model_precision = None

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(train_data.drop('LUNG_CANCER', axis=1), train_data['LUNG_CANCER']):
        X_train, X_val = train_data.drop('LUNG_CANCER', axis=1).iloc[train_idx], train_data.drop('LUNG_CANCER', axis=1).iloc[val_idx]
        y_train, y_val = train_data['LUNG_CANCER'].iloc[train_idx], train_data['LUNG_CANCER'].iloc[val_idx]

        # Calculate scale_pos_weight dynamically for each fold's distribution
        scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=scale_weight,
            learning_rate=0.05,
            max_depth=4,
            n_estimators=100,
            subsample=0.8,
            random_state=42,
            eval_metric='auc'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        # Evaluate on the validation fold
        val_preds_proba = model.predict_proba(X_val)[:, 1]
        val_preds = model.predict(X_val)

        fold_auc = roc_auc_score(y_val, val_preds_proba)
        fold_accuracy = accuracy_score(y_val, val_preds)
        fold_recall = recall_score(y_val, val_preds)
        fold_f1 = f1_score(y_val, val_preds)
        fold_precision = precision_score(y_val, val_preds)

        auc_scores.append(fold_auc)
        f1_scores.append(fold_f1)
        accuracy_scores.append(fold_accuracy)
        recall_scores.append(fold_recall)
        precision_scores.append(fold_precision)

        print(f"Validation ROC-AUC: {fold_auc:.4f}")
        print(f"Validation Accuracy: {fold_accuracy:.4f}, Recall: {fold_recall:.4f}, F1-Score: {fold_f1:.4f}, Precision: {fold_precision:.4f}")

        # Track the best model across folds
        if fold_auc > best_auc:
            best_auc = fold_auc
            best_model_auc = model
        if fold_f1 > best_f1:
            best_f1 = fold_f1
            best_model_f1 = model
        if fold_recall > best_recall:
            best_recall = fold_recall
            best_model_recall = model  
        if fold_accuracy > best_accuracy:
            best_accuracy = fold_accuracy
            best_model_accuracy = model
        if fold_precision > best_precision:
            best_precision = fold_precision
            best_model_precision = model
            
    print(f"\nAverage Validation ROC-AUC across 5 folds: {np.mean(auc_scores):.4f}")
    print(f"Average Validation Accuracy across 5 folds: {np.mean(accuracy_scores):.4f}")
    print(f"Average Validation Recall across 5 folds: {np.mean(recall_scores):.4f}")
    print(f"Average Validation F1-Score across 5 folds: {np.mean(f1_scores):.4f}")
    print(f"Average Validation Precision across 5 folds: {np.mean(precision_scores):.4f}")

    os.makedirs("src/models/XGBoost/trained", exist_ok=True)
    os.makedirs("src/models/XGBoost/saved_and_tested", exist_ok=True)

    xgb_result = pd.DataFrame({
        'Fold': list(range(1, 6)),
        'ROC-AUC': auc_scores,
        'Accuracy': accuracy_scores,
        'Recall': recall_scores,
        'F1-Score': f1_scores,
        'Precision': precision_scores
    })
    xgb_result.to_csv("src/models/XGBoost/trained_and_tested/xgb_kfold_results.csv", index=False)

    # Save the highest performing model for the final test
    best_model_auc.save_model("src/models/XGBoost/saved/best_xgb_auc_lung_cancer.json")
    best_model_f1.save_model("src/models/XGBoost/saved/best_xgb_f1_lung_cancer.json")
    best_model_recall.save_model("src/models/XGBoost/saved/best_xgb_recall_lung_cancer.json")
    best_model_accuracy.save_model("src/models/XGBoost/saved/best_xgb_accuracy_lung_cancer.json")
    best_model_precision.save_model("src/models/XGBoost/saved/best_xgb_precision_lung_cancer.json")
    print("Best model saved to 'src/models/XGBoost/saved/'.")

if __name__ == "__main__":
    run_kfold_training()