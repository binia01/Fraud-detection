# src/preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from src.utils import ip_to_int

class BasePreprocessor:
    def __init__(self, df):
        self.df = df

    def check_missing(self):
        print("Missing Values:\n", self.df.isnull().sum())

    def clean_duplicates(self):
        initial = self.df.shape[0]
        self.df.drop_duplicates(inplace=True)
        print(f"Dropped {initial - self.df.shape[0]} duplicate rows.")
        
    def separate_features_target(self, target_col):
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        return X, y

    def handle_imbalance_smote(self, X, y):
        """
        Applies SMOTE. 
        Note: Expected to be used on Training data ONLY in the modeling phase.
        """
        print("Original class distribution:", y.value_counts())
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        print("Resampled class distribution:", y_res.value_counts())
        return X_res, y_res

class EcommercePreprocessor(BasePreprocessor):
    def __init__(self, df, ip_country_df):
        super().__init__(df)
        self.ip_country_df = ip_country_df

    def fix_datetimes(self):
        """Convert time columns to datetime objects."""
        print("Converting timestamps...")
        self.df['signup_time'] = pd.to_datetime(self.df['signup_time'])
        self.df['purchase_time'] = pd.to_datetime(self.df['purchase_time'])

    def feature_engineering(self):
        """Create time-based and velocity features."""
        print("Engineering features...")
        
        # Time-based
        self.df['hour_of_day'] = self.df['purchase_time'].dt.hour
        self.df['day_of_week'] = self.df['purchase_time'].dt.dayofweek
        
        # Time since signup (in seconds)
        self.df['time_since_signup'] = (self.df['purchase_time'] - self.df['signup_time']).dt.total_seconds()
        
        # Transaction Velocity (Simplified: count purchases per user)
        # In a real streaming scenario, this would be complex. Here we aggregate.
        self.df['transaction_count'] = self.df.groupby('user_id')['user_id'].transform('count')

    def merge_geolocation(self):
        """
        Merges IP data. This is computationally expensive. 
        """
        print("Processing Geolocation (this may take a while)...")
        
        # 1. Convert IP to int using the updated utility
        self.df['ip_int'] = ip_to_int(self.df['ip_address'])
        
        # 2. Optimized Range Merge
        # Ensure the IP table is sorted by lower_bound
        self.ip_country_df = self.ip_country_df.sort_values('lower_bound_ip_address')
        
        ip_low = self.ip_country_df['lower_bound_ip_address'].values
        ip_high = self.ip_country_df['upper_bound_ip_address'].values
        country_names = self.ip_country_df['country'].values
        
        countries = []
        ip_ints = self.df['ip_int'].values
        
        for ip in ip_ints:
            # Binary search for the index where ip fits
            idx = np.searchsorted(ip_low, ip, side='right') - 1
            
            # Check if it actually falls within the range [lower, upper]
            if idx >= 0 and ip <= ip_high[idx]:
                countries.append(country_names[idx])
            else:
                countries.append("Unknown")
                
        self.df['country'] = countries
        
        # Debugging print to confirm it worked
        print(f"Geolocation finished. Found {len(set(countries))} unique countries.")

    def encode_categoricals(self):
        print("Encoding categoricals...")
        # One-Hot Encoding for low cardinality
        self.df = pd.get_dummies(self.df, columns=['source', 'browser', 'sex', 'country'], drop_first=True)
        # Drop columns not needed for modeling
        self.df.drop(columns=['user_id', 'device_id', 'ip_address', 'signup_time', 'purchase_time', 'ip_int'], inplace=True, errors='ignore')

    def scale_features(self):
        print("Scaling numerical features...")
        scaler = MinMaxScaler() # Or StandardScaler
        cols_to_scale = ['purchase_value', 'age', 'time_since_signup', 'transaction_count']
        self.df[cols_to_scale] = scaler.fit_transform(self.df[cols_to_scale])


class CreditCardPreprocessor(BasePreprocessor):
    def scale_amount_time(self):
        print("Scaling Amount and Time...")
        scaler = StandardScaler()
        self.df['normalized_amount'] = scaler.fit_transform(self.df['Amount'].values.reshape(-1, 1))
        # Time is seconds from start, scaling helps convergence
        self.df['normalized_time'] = scaler.fit_transform(self.df['Time'].values.reshape(-1, 1))
        self.df.drop(['Time', 'Amount'], axis=1, inplace=True)