import pandas as pd
from sklearn.model_selection import train_test_split
import os
import warnings

class DataPreprocessor:
    def __init__(self, input_path, output_dir):
        self.input_path = input_path
        self.output_dir = output_dir

    def load_data(self):
        """1. Load the dataset from the given filepath."""
        df = pd.read_csv(self.input_path)
        # Strip whitespace from column names right away to prevent downstream errors
        df.columns = df.columns.str.strip()
        return df

    def validate_schema(self, df):
        """2. Validate expected columns, target presence, and basic data types."""
        expected_columns = [
            'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 
            'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 
            'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 
            'SWALLOWING DIFFICULTY', 'CHEST PAIN', 'LUNG_CANCER'
        ]
        
        # Check expected columns
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")
            
        # Check target
        if 'LUNG_CANCER' not in df.columns:
            raise ValueError("Target column 'LUNG_CANCER' is missing.")
            
        # Check basic types (ensure AGE is numerical)
        if not pd.api.types.is_numeric_dtype(df['AGE']):
            warnings.warn("AGE column is not numeric. Attempting conversion.")
            df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce')
            
        return df

    # Wrong
    # When handling missing values, different strategies can be applied but branches per model
    # Fix by making sklearn's models in question handle missing values natively
    # The idea is still letting the model learn from missing values instead of deletion.
    # def check_missing_values(self, df):
    #     """3. Identify and handle missing values."""
    #     missing_counts = df.isnull().sum()
    #     if missing_counts.sum() > 0:
    #         warnings.warn(f"Missing values detected:\n{missing_counts[missing_counts > 0]}")
    #         # Dropping for simplicity in this baseline; imputation could be added here
    #         df = df.dropna().reset_index(drop=True)
    #     return df

    def check_duplicates(self, df):
        """4. Identify and remove duplicate rows."""
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df = df.drop_duplicates().reset_index(drop=True)
        return df

    def validate_values(self, df):
        """5. Validate numerical ranges and categorical consistency."""
        # Check AGE range (e.g., 0 to 120)
        invalid_ages = df[(df['AGE'] < 0) | (df['AGE'] > 120)]
        if not invalid_ages.empty:
            warnings.warn(f"Found {len(invalid_ages)} rows with impossible ages. Dropping them.")
            df = df[(df['AGE'] >= 0) & (df['AGE'] <= 120)].copy()

        # Check valid values in target and gender
        if not set(df['LUNG_CANCER'].unique()).issubset({'YES', 'NO'}):
            warnings.warn("Unexpected values found in LUNG_CANCER column.")
        
        if not set(df['GENDER'].unique()).issubset({'M', 'F'}):
            warnings.warn("Unexpected values found in GENDER column.")
            
        return df

    # Not neccessary for xgboost and random forest abstractly but this shit is real for the implementation of scikit learn for some reason
    def encode_features(self, df):
        """6. Encode categorical features and target to 1/0 binary format."""
        df = df.copy()
        
        # Target and Gender
        df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
        df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})
        
        # The survey encodes symptoms as 1 (NO) and 2 (YES). Map to 0 and 1.
        symptom_columns = [
            'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
            'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 
            'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 
            'SWALLOWING DIFFICULTY', 'CHEST PAIN'
        ]
        
        for col in symptom_columns:
            df[col] = df[col].map({1: 0, 2: 1})
            
        return df

    # def remove_irrelevant_columns(self, df):
    #     """7. Remove any accidental indices, IDs, or leakage candidates."""
    #     # This dataset specifically doesn't have IDs, but this serves as a safeguard
    #     # However, if there are columns that were ids, i would still leave it to do gain ratio or something similar.
    #     cols_to_drop = [col for col in df.columns if 'ID' in col.upper() or 'INDEX' in col.upper()]
    #     if cols_to_drop:
    #         df = df.drop(columns=cols_to_drop)
    #     return df

    def split_features_target(self, df):
        """8. Separate into feature matrix X and target vector y."""
        X = df.drop('LUNG_CANCER', axis=1)
        y = df['LUNG_CANCER']
        return X, y

    def split_train_test(self, X, y, test_size=0.2, random_state=42):
        """9. Perform stratified train-test split."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        train_path = os.path.join(self.output_dir, "train.csv")
        test_path = os.path.join(self.output_dir, "test.csv")
        
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

    def run_preprocessing_pipeline(self):
        """Executes the entire pipeline sequentially."""
        df = self.load_data()
        df = self.validate_schema(df)
        # df = self.check_missing_values(df)
        df = self.check_duplicates(df)
        df = self.validate_values(df)
        df = self.encode_features(df)
        # df = self.remove_irrelevant_columns(df)
        
        X, y = self.split_features_target(df)
        self.split_train_test(X, y)

if __name__ == "__main__":
    processor = DataPreprocessor(
        input_path = "data/raw/survey_lung_cancer.csv",
        output_dir = "data/processed"
    )

    processor.run_preprocessing_pipeline()