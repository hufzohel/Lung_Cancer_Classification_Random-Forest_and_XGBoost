import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

def run_kfold_training():
    train_data = pd.read_csv("data/processed/train.csv")
    auc_scores = []
    best_auc = 0

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
        fold_auc = roc_auc_score(y_val, val_preds_proba)
        auc_scores.append(fold_auc)
        
        print(f"Fold {fold} - Validation ROC-AUC: {fold_auc:.4f}")
        
        # Track the best model across folds
        if fold_auc > best_auc:
            best_auc = fold_auc
            best_model = model
            
    print(f"\nAverage Validation ROC-AUC across 5 folds: {np.mean(auc_scores):.4f}")
    
    # Save the highest performing model for the final test
    best_model.save_model("best_xgb_lung_cancer.json")
    print("Best model saved to 'best_xgb_lung_cancer.json'.")

if __name__ == "__main__":
    run_kfold_training()