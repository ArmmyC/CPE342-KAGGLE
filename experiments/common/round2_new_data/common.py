import pandas as pd, numpy as np
import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D={1:'01_binary_classification',2:'02_multiclass_classification',3:'03_regression',5:'05_anomaly_detection'}
TARGET={1:'is_cheater',2:'segment',3:'spending_30d'}
def load(k):
    tr=pd.read_csv(f'{ROOT}/{D[k]}/train.csv') if k!=5 else None
    te=pd.read_csv(f'{ROOT}/{D[k]}/test.csv')
    return tr,te
