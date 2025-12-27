import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report,
    precision_recall_curve
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

class ModelTrainer:
    def __init__(self, df: pd.DataFrame, target_col):
        self.df = df
        self.target_col = target_col
        self.X = df.drop(columns=[target_col])
        self.y = df[target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = (None, None, None, None)
        self.models = {}
        self.results = []

    def split_data(self, test_size=0.2, random_state=42):
        '''Split data into training and test sets using Stratified Split'''
        print(f"Splitting data (Test size: {test_size})....")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size= test_size, random_state=random_state, stratify=self.y)
        print(f"Train Shape: {self.X_train.shape}, Test Shape: {self.X_test.shape}")

    
    def train_with_cv(self, model, model_name, k=5):
        """
        Trains model using Stratified K-Fold CV with SMOTE inside the pipeline.
        This prevents data leakage where synthetic test data is seen during training.
        """
        print(f"\n--- Training {model_name} with Stratified K-Fold (k={k}) ---")

        pipeline = ImbPipeline(steps=[
            ('smote', SMOTE(random_state=42)),
            ('model', model)
        ])

        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        f1_scores = cross_val_score(pipeline, self.X_train, self.y_train, cv=cv, scoring='f1')

        print(f"Mean F1-Score (CV): {f1_scores.mean():.4f} (+/- {f1_scores.std():.4f})")
        
        # Fit on the full training set for final evaluation
        pipeline.fit(self.X_train, self.y_train)
        self.models[model_name] = pipeline
        
        return pipeline
    
    def evaluate_model(self, model_name):
        if model_name not in self.models:
            print(f"Error: {model_name} not trained yet.")

        model = self.models[model_name]

         # Predictions
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1] if hasattr(model.named_steps['model'], "predict_proba") else y_pred

        # Metrics
        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred)
        rec = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_prob)
        pr_auc = average_precision_score(self.y_test, y_prob) # Area Under Precision-Recall Curve

        print(f"\n--- Test Set Results: {model_name} ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC AUC:   {roc_auc:.4f}")
        print(f"PR AUC:    {pr_auc:.4f} (Key Metric for Imbalance)")
        
        # Save results for comparison
        self.results.append({
            'Model': model_name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'ROC AUC': roc_auc,
            'PR AUC': pr_auc
        })

        # Visualizations
        self._plot_confusion_matrix(self.y_test, y_pred, model_name)
        self._plot_pr_curve(self.y_test, y_prob, model_name)

    def _plot_confusion_matrix(self, y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {title}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

    def _plot_pr_curve(self, y_true, y_prob, title):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.figure(figsize=(6, 4))
        plt.plot(recall, precision, marker='.', label=title)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {title}')
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_comparison_table(self):
        return pd.DataFrame(self.results).sort_values(by='PR AUC', ascending=False)