import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Read data from CSV
df_all = pd.read_csv('all_data.csv')

# Separate features and labels
X_all = df_all.drop('label', axis=1)
y_all = df_all['label']

# Standardize the data
scaler_all = StandardScaler()
X_std_all = scaler_all.fit_transform(X_all)

# Perform PCA before removing outliers
pca_all = PCA(n_components=2)
X_pca_all = pca_all.fit_transform(X_std_all)

# Plot the PCA results before removing outliers
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_pca_all[:, 0], X_pca_all[:, 1], c=y_all, cmap='viridis')
plt.title('PCA Results Before Removing Outliers')

# Define a threshold to identify outliers (you can adjust this based on your data)
threshold = 8   # Values greater than threshold will be removed

# Identify and remove outliers
mask_all = np.any(np.abs(X_all) > threshold, axis=1)
X_no_outliers_all = X_all[~mask_all]
y_no_outliers_all = y_all[~mask_all]

# Export outliers to CSV
df_outliers_all = df_all[mask_all]
df_outliers_all.to_csv('outliers_all.csv', index=False)

# Separate features and labels for data without outliers
X_no_outliers_all_df = pd.DataFrame(data=X_no_outliers_all, columns=['feature1', 'feature2'])
X_no_outliers_all_df['label'] = y_no_outliers_all.astype(int)

# Standardize the data without outliers
scaler_no_outliers_all = StandardScaler()
X_no_outliers_std_all = scaler_no_outliers_all.fit_transform(X_no_outliers_all_df.drop('label', axis=1))

# Perform PCA after removing outliers
pca_no_outliers_all = PCA(n_components=2)
X_pca_no_outliers_all = pca_no_outliers_all.fit_transform(X_no_outliers_std_all)

# Plot the PCA results after removing outliers
plt.subplot(1, 2, 2)
plt.scatter(X_pca_no_outliers_all[:, 0], X_pca_no_outliers_all[:, 1], c=y_no_outliers_all, cmap='viridis')
plt.title('PCA Results After Removing Outliers')

plt.show()
