import pandas as pd
import numpy as np
from MSCVAE import MSCVAE

# 1. Create Synthetic Data
print("Generating synthetic data...")
timestamps = pd.date_range(start="2023-01-01", periods=1000, freq="1min")
data = np.random.randn(1000, 5) # 5 features
cols = [f"sensor_{i}" for i in range(5)]
df_train = pd.DataFrame(data, columns=cols, index=timestamps)

# Inject anomaly in test set
data_test = np.random.randn(200, 5)
data_test[100:110, 0] += 5 # Anomaly in sensor_0
timestamps_test = pd.date_range(start="2023-01-02", periods=200, freq="1min")
df_test = pd.DataFrame(data_test, columns=cols, index=timestamps_test)

# 2. Instantiate and Fit
print("\nInstantiating MSCVAE...")
model = MSCVAE(n_features=5, window_size=10, stride=1)

print("Fitting model...")
model.fit(df_train, epochs=2, verbose=True) # Low epochs for speed

# 3. Predict
print("\nPredicting on test set...")
# Pass timestamps explicitly although index is also available
predictions = model.predict(df_test, timestamps=df_test.index)

print("Prediction results (first 5):")
print(pd.DataFrame(predictions).head())

# 4. Contribution Analysis
print("\nCalculating contributions...")
contrib_dict, recon_df = model.contribution(df_test)

print("Contribution Dictionary:")
print(contrib_dict)

print("\nReconstructed DataFrame (head):")
print(recon_df.head())

print("\nExample script completed successfully.")
