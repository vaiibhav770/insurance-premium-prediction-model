import pickle
import pandas as pd
with open('Model/model.pkl', 'rb') as f:
    model = pickle.load(f)

MODEL_VERSION= '1.0.0'