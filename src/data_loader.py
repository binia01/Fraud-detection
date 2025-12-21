import pandas as pd
import os

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_csv(self, filename):
        full_path = os.path.join(self.data_path, filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not dound: {full_path}")
        
        print(f"Loading {filename}...")
        return pd.read_csv(full_path)

    