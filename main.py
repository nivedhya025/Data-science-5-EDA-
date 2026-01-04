import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

import plotly.express as px

# Load dataset
df = pd.read_csv("uae_real_estate_2024.csv")

#--------------------------------
# #STEP 1: IMPORT Data
# ------------------------------

print("\n First 5 rows of the dataset: \n")
print(df.head())

print("\n Dataset Info: \n")
print(df.info())

#--------------------------------
# STEP 2: EXPORT Data
# ------------------------------

print("\n Exporting cleaned dataset to 'uae_real_estate.csv'\n")
df.to_csv("uae_real_estate.csv", index=False)
print("\n Exporting cleaned dataset to 'uae_real_estate.xlsx'\n")
df.to_excel("uae_real_estate.xlsx", index=False)

print("\n Data exported successfully. \n")

#--------------------------------
# STEP 3: DATA CLEANING
# ------------------------------

# -----------------------------
# MISSING VALUES
# -----------------------------
print("Missing values BEFORE handling:")
print(df.isnull().sum())

# numeric columns → mean
numeric_cols = df.select_dtypes(include="number").columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

# categorical columns → mode
categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values AFTER handling:")
print(df.isnull().sum())

# -----------------------------
# DUPLICATES
# -----------------------------
print("\nDuplicate rows BEFORE removal:")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("Duplicate rows AFTER removal:")
print(df.duplicated().sum())

# -----------------------------
# OUTLIERS (IQR on PRICE if exists)
# -----------------------------
if "price" in df.columns:
    Q1 = df["price"].quantile(0.25)
    Q3 = df["price"].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    print("\nOutliers BEFORE handling:")
    print(df[(df["price"] < lower) | (df["price"] > upper)].shape[0])

    df = df[(df["price"] >= lower) & (df["price"] <= upper)]

    print("Outliers AFTER handling:")
    print(df[(df["price"] < lower) | (df["price"] > upper)].shape[0])

# -----------------------------
# SAVE CLEANED DATA
# -----------------------------
df.to_csv("cleaned_uae_real_estate_2024.csv", index=False)

print("\n Data cleaning completed successfully")

#--------------------------------
# STEP 4: DATA TRANSFORMATION
# ------------------------------

# ============================================================
# DATA TRANSFORMATION – NORMALIZATION / STANDARDIZATION
# ============================================================

# Load dataset
df = pd.read_csv("uae_real_estate_2024.csv")

# 1. Select numerical features for transformation
num_features = df.select_dtypes(include=np.number).columns

# 2. Standardization (Z-score scaling)

Mean = 0, 
StandardDeviation = 1
scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])
print("\n Data standardization completed successfully.\n")
print(df[num_features].head())

# ---- OR ----
# If normalization is required instead of standardization,
# uncomment the code below and comment the StandardScaler section

scaler = MinMaxScaler()
df[num_features] = scaler.fit_transform(df[num_features])

# 3. Verify transformed data

print("\n Data normalization completed successfully.\n")    
print(df[num_features].head())
print(df[num_features].describe())


#--------------------------------
# STEP 5: DESCRIPTIVE STATISTICS            
# ------------------------------

# Step 1: Load the dataset
df = pd.read_csv("uae_real_estate_2024.csv")

# Step 5: Descriptive Statistics
print("Descriptive Statistics:\n")
print(df.describe())

# Optional: Individual statistics (for clarity)
print("\nMean:\n", df.mean(numeric_only=True))
print("\nMedian:\n", df.median(numeric_only=True))
print("\nMode:\n", df.mode(numeric_only=True).iloc[0])
print("\nStandard Deviation:\n", df.std(numeric_only=True))
print("\nMinimum:\n", df.min(numeric_only=True))
print("\nMaximum:\n", df.max(numeric_only=True))


# ============================================================
# STEP 6: BASIC VISUALIZATION
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("uae_real_estate_2024.csv")

# -------------------------------
# Line Plot (example: price trend)
# -------------------------------
plt.figure()
df['price'].head(50).plot()
plt.title("Property Price Trend")
plt.xlabel("Index")
plt.ylabel("Price")
plt.show()

# -------------------------------
# Bar chart: Average price by number of bedrooms
# -------------------------------

df.groupby('bedrooms')['price'].mean().plot(kind='bar')
plt.title("Average Property Price by Bedrooms")
plt.xlabel("Number of Bedrooms")
plt.ylabel("Average Price")
plt.show()
plt.figure()


# -------------------------------
# Histogram (price distribution)
# -------------------------------
plt.hist(df['price'], bins=40, color='skyblue', edgecolor='black')
plt.title("Distribution of Property Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# ============================================================
# STEP 7: ADVANCED VISUALIZATION 
# ============================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("uae_real_estate_2024.csv")

# Select numeric columns only
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# -------------------------------
# Pair Plot
# -------------------------------
sns.pairplot(numeric_df)
plt.show()

# -------------------------------
# Heatmap (Correlation)
# -------------------------------

# Select numeric columns
numeric_df = df.select_dtypes(include='number')

# Correlation matrix
corr = numeric_df.corr()

# Plot colored heatmap
plt.figure(figsize=(9,7))
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",   # color gradient
    linewidths=0.5,
    cbar=True,         # show color bar
    vmin=-1, vmax=1
)

plt.title("Correlation Heatmap of UAE Real Estate Data")
plt.show()


# -------------------------------
# Violin Plot (example: price vs bedrooms)
# -------------------------------
sns.violinplot(x='type', y='price', data=df)
plt.xticks(rotation=45)
plt.title("Price Distribution by Property Type")
plt.show()
# -------------------------------


# Select numeric features
num_df = df.select_dtypes(include=['int64','float64'])

# Correlation matrix
correlation = num_df.corr()
print("Correlation Matrix:\n", correlation)

# Covariance matrix
covariance = num_df.cov()
print("\nCovariance Matrix:\n", covariance)


#--------------------------------
# STEP 8: INTERACTIVE VISUALIZATION
# ------------------------------

import pandas as pd
import plotly.express as px

# Load dataset
df = pd.read_csv("uae_real_estate_2024.csv")

# Create interactive scatter plot
fig = px.scatter(df,x="sizeMin",y="price",
    color="type",
    title="Interactive Property Price vs Area",
    labels={
        "sizeMin": "Property Size (sq ft)",
        "price": "Price",
        "type": "Property Type"
    }
)

# Improve layout (similar to your output)
fig.update_layout(
    xaxis_title="sizeMin",
    yaxis_title="price",
    template="plotly_white"
)

fig.show()

#--------------------------------
# STEP 9: PROBABILITY ANALYSIS 
# ------------------------------


# Load dataset
df = pd.read_csv("uae_real_estate_2024.csv")
sns.histplot(df['price'], kde=True)
plt.title("Probability Distribution of Property Prices")
plt.show()

#--------------------------------
# STEP 10: MODELING (K-NN CLASSIFICATION)       
# ------------------------------


numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns   

df['price_category'] = pd.qcut(df['price'], 3, labels=['Low', 'Medium', 'High'])
Scaler = StandardScaler()
df.scaled = Scaler.fit_transform(df[numeric_cols])
df.scaled = pd.DataFrame(df.scaled, columns=numeric_cols)   
X = df.scaled[numeric_cols]
y = df['price_category']      

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("\n k-NN Accuracy: \n", accuracy_score(y_test, y_pred))


#--------------------------------
# STEP 11: MODELING (K-MEANS CLUSTERING)            
# ------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("uae_real_estate_2024.csv")

# Clean sizeMin column (remove 'sqft')
df["sizeMin"] = df["sizeMin"].astype(str).str.replace("sqft", "", regex=False)
df["sizeMin"] = pd.to_numeric(df["sizeMin"], errors="coerce")

# Drop rows with missing values
df = df.dropna(subset=["sizeMin", "price"])

# Select numeric features
X = df[["sizeMin", "price"]]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply k-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Plot
plt.figure(figsize=(8,6))
plt.scatter(df["sizeMin"], df["price"], c=df["Cluster"])
plt.title("K-Means Clustering of Properties")
plt.xlabel("Property Size")
plt.ylabel("Price")
plt.show()

#--------------------------------
# STEP 12: SUMMARY AND INSIGHTS
# ------------------------------

print("""\n
SUMMARY & INSIGHTS:
1. Property prices increase with property size.
2. Apartments dominate the listings.
3. Strong positive correlation exists between size and price.
4. k-NN effectively classified properties into price segments.
5. K-Means identified three distinct market clusters.
""")
