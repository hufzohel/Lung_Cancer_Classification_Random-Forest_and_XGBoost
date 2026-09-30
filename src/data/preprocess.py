import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

class LungCancerDataLoader:
    def __init__(self, filepath="survey lung cancer.csv", test_size=0.2, random_state=42):
        self.filepath = filepath
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self._prepare_base_data()

    def _prepare_base_data(self):
        """Cleans data and isolates the holdout test set."""
        df = pd.read_csv(self.filepath)
        df.columns = df.columns.str.strip()
        df = df.drop_duplicates().reset_index(drop_index=True)
        
        symptom_columns = [
            'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
            'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 
            'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 
            'SWALLOWING DIFFICULTY', 'CHEST PAIN'
        ]
        
        for col in symptom_columns:
            df[col] = df[col].map({1: 0, 2: 1})
            
        df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})
        df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
        
        X = df.drop('LUNG_CANCER', axis=1)
        y = df['LUNG_CANCER']
        
        # Split off the final test set immediately (Stratified)
        self.X_train_full, self.X_test_raw, self.y_train_full, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

    def load_kfold_train_val(self, n_splits=5):
        """Yields standardized train and validation sets for each K-Fold split."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        # Reset index to allow iloc slicing during K-Fold
        X_base = self.X_train_full.reset_index(drop=True)
        y_base = self.y_train_full.reset_index(drop=True)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_base, y_base), 1):
            X_fold_train, X_fold_val = X_base.iloc[train_idx].copy(), X_base.iloc[val_idx].copy()
            y_fold_train, y_fold_val = y_base.iloc[train_idx], y_base.iloc[val_idx]
            
            # Fit scaler ONLY on this fold's training data to prevent data leakage
            X_fold_train['AGE'] = self.scaler.fit_transform(X_fold_train[['AGE']])
            X_fold_val['AGE'] = self.scaler.transform(X_fold_val[['AGE']])
            
            yield fold, X_fold_train, y_fold_train, X_fold_val, y_fold_val

    def load_test_set(self):
        """Returns the fully scaled holdout test set."""
        # Fit scaler on the entire training set, transform the test set
        self.scaler.fit(self.X_train_full[['AGE']])
        
        X_test_scaled = self.X_test_raw.copy()
        X_test_scaled['AGE'] = self.scaler.transform(X_test_scaled[['AGE']])
        
        return X_test_scaled, self.y_test