import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv(
   r"C:\Users\hp\PycharmProjects\Placementpredictionsystem\data\placement_data (1).csv"
)

# Show actual columns
print("\nActual columns in dataset:")
for column in df.columns:
    print(repr(column))

# Select features
features = [
    "CGPA",
    "AttendencePercent",
    "Internships",
    "Project",
    "coding test score"
]

print("\nSelected Features:")
print(features)

# Check which features are missing
missing_features = [f for f in features if f not in df.columns]

if missing_features:
    print("\nERROR: These columns were not found:")
    print(missing_features)
    print("\nPlease use the exact column names printed above.")
    exit()

# Create X
X = df[features]

# Standardize
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)

# K-Means
kmeans = KMeans(
    n_clusters=2,
    random_state=0,
    n_init=10
)

clusters = kmeans.fit_predict(x_scaled)

# Add cluster
df["Cluster"] = clusters

print("\nCluster Assignment:")
print(df[features + ["Cluster"]].head(10))

# Cluster centers
centres = kmeans.cluster_centers_

# Convert back to original scale
center = scaler.inverse_transform(centres)

center_df = pd.DataFrame(
    center,
    columns=features
)

center_df.index.name = "Cluster"

print("\nCluster Centers:")
print(center_df)

# Plot
plt.figure(figsize=(8, 6))

plt.scatter(
    x_scaled[:, 0],
    x_scaled[:, 1],
    c=clusters
)


plt.xlabel("CGPA")
plt.ylabel("Attendance Percentage")
plt.title("K-Means Clustering")

plt.show()