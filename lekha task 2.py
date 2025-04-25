import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv(r"C:\Users\lekha\Downloads\task2.csv")

print("Dataset Summary Statistics:")
print(data.describe())

numeric_features = data.select_dtypes(include=[np.number]).columns

plt.figure(figsize=(12, 10))
for i, column in enumerate(numeric_features, 1):
    plt.subplot(3, 3, i) 
    sns.histplot(data[column], kde=True)
    plt.title(f"Histogram of {column}")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 10))
for i, column in enumerate(numeric_features, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(data[column])
    plt.title(f"Boxplot of {column}")
plt.tight_layout()
plt.show()

sns.pairplot(data[numeric_features])
plt.suptitle("Pairplot of Numeric Features", y=1.02)
plt.show()

corr_matrix = data[numeric_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()