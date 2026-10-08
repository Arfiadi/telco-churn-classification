import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer untuk domain feature engineering dataset Telco Churn.
    Mengekstrak:
    - TotalServices: Jumlah layanan bernilai tambah aktif (switching barrier).
    - AverageMonthlyCost: Total pengeluaran per tenure (mengatasi interaksi non-linier).
    - MonthlyChargesRatio: Rasio beban berjalan terhadap historis (deteksi bill-shock).
    - TenureGroup: Kohort masa langganan (New, Growing, Mature, Loyal).
    """
    def __init__(self):
        self.service_cols = [
            'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_out = X.copy()
        
        # 1. Total layanan aktif
        services_count = np.zeros(len(X_out))
        for col in self.service_cols:
            if col in X_out.columns:
                services_count += (X_out[col] == 'Yes').astype(int)
        X_out['TotalServices'] = services_count
        
        # 2. Rata-rata biaya bulanan & rasio fluktuasi tagihan
        tenure_safe = np.where(X_out['tenure'] == 0, 1, X_out['tenure'])
        avg_monthly = X_out['TotalCharges'] / tenure_safe
        avg_monthly = avg_monthly.fillna(X_out['MonthlyCharges'])
        X_out['AverageMonthlyCost'] = avg_monthly
        
        safe_avg = np.where(X_out['AverageMonthlyCost'] == 0, 1, X_out['AverageMonthlyCost'])
        X_out['MonthlyChargesRatio'] = X_out['MonthlyCharges'] / safe_avg
        
        # 3. Kohort masa langganan
        tenure_bins = [-1, 12, 24, 48, 100]
        tenure_labels = ['New', 'Growing', 'Mature', 'Loyal']
        X_out['TenureGroup'] = pd.cut(X_out['tenure'], bins=tenure_bins, labels=tenure_labels).astype(str)
        
        return X_out
