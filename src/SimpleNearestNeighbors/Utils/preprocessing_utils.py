from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold

"""
If to_fit == True we assume that scaler is already fitted
because we might want to normalize test data, thus we want
to scale them with a scaler fitted to training data
"""
def normalize_dataset(instances, scaler=MinMaxScaler(), to_fit=True):
    if to_fit:
        scaler.fit(instances)
    
    new_instances = scaler.transform(instances)

    return scaler, new_instances
