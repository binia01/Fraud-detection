import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

class ModelExplainer:
    def __init__(self, model, X_test):
        self.model = model
        self.X_test = X_test
        
        # 1. Extract the actual model from the Pipeline
        if hasattr(model, 'named_steps'):
            self.estimator = model.named_steps['model']
        else:
            self.estimator = model
            
        # 2. Initialize SHAP Explainer
        # TreeExplainer is fast for RF/XGBoost.
        self.explainer = shap.TreeExplainer(self.estimator)
        
        # 3. Optimize: Calculate SHAP values for a small sample
        self.sample_size = min(200, len(X_test))
        self.X_sample = shap.sample(X_test, self.sample_size)
        
        print(f"Calculating SHAP values for a sample of {self.sample_size} rows...")
        self.shap_values_sample = self.explainer.shap_values(self.X_sample)
        print("SHAP values calculated.")

    def plot_feature_importance(self, top_n=10):
        """Plots the built-in feature importance from the model."""
        if hasattr(self.estimator, 'feature_importances_'):
            importances = self.estimator.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            plt.figure(figsize=(10, 6))
            plt.title(f"Top {top_n} Feature Importances (Built-in)")
            plt.bar(range(top_n), importances[indices], align="center")
            plt.xticks(range(top_n), self.X_test.columns[indices], rotation=45, ha='right')
            plt.xlim([-1, top_n])
            plt.tight_layout()
            plt.show()
        else:
            print("Model does not support built-in feature importance.")

    def plot_shap_summary(self):
        """Generates the SHAP Summary Plot using the sample data."""
        print("Generating SHAP Summary Plot...")
        
        # Handle binary classification (SHAP returns list [class0, class1])
        vals_to_plot = self.shap_values_sample
        if isinstance(vals_to_plot, list):
            vals_to_plot = vals_to_plot[1] # Class 1 (Fraud)

        # Handle 3D Interaction Values (Flatten if needed)
        if len(np.array(vals_to_plot).shape) == 3:
            vals_to_plot = np.abs(vals_to_plot).sum(axis=2)

        shap.summary_plot(vals_to_plot, self.X_sample)

    def plot_shap_force(self, index, matplotlib=True):
        """
        Generates a Force Plot for a SPECIFIC row index from the FULL X_test.
        """
        print(f"Generating Force Plot for index {index}...")
        
        # 1. Get the specific row data
        row_df = self.X_test.iloc[[index]] 
        features = self.X_test.iloc[index]
        
        # 2. Calculate SHAP just for this row
        shap_val_raw = self.explainer.shap_values(row_df)
        
        # 3. Extract Correct Values (Handling Binary/Arrays)
        # We always aim for Class 1 (Fraud)
        
        # Handle SHAP Values
        if isinstance(shap_val_raw, list):
            # Binary: [Class0, Class1]
            shap_values = shap_val_raw[1][0]
        else:
            # Single Output: (1, features)
            shap_values = shap_val_raw[0]

        # Handle Base Value (Expected Value)
        base_val_raw = self.explainer.expected_value
        
        # If expected_value is an array/list (e.g. [0.9, 0.1]), pick the second one for Class 1
        if hasattr(base_val_raw, '__len__') and len(base_val_raw) > 1:
            base_value = base_val_raw[1]
        elif hasattr(base_val_raw, '__len__') and len(base_val_raw) == 1:
            base_value = base_val_raw[0]
        else:
            base_value = base_val_raw
            
        # Ensure it's a python float (scalar)
        if isinstance(base_value, np.ndarray):
            base_value = base_value.item()

        # 4. Handle 3D Interaction edge case for single row
        if len(shap_values.shape) == 2: # (features, features)
             shap_values = shap_values.sum(axis=1)

        if matplotlib:
            plt.figure(figsize=(22,5))


        # 5. Plot
        return shap.force_plot(
            base_value, 
            shap_values, 
            features, 
            matplotlib=matplotlib,
            show=True,
            text_rotation=45
        )