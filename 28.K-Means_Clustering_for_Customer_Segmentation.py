import pandas as pd
from sklearn.cluster import KMeans

# Load the customer dataset (replace 'customer_data.csv' with your dataset)
customer_data = pd.read_csv('customer_data.csv')

# Preprocess the data as needed (e.g., handle missing values, scaling)

# Determine the number of clusters (K)
# You can use the Elbow Method or other methods to find the optimal K
k = 3

# Create a K-Means clustering model
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(customer_data)

# Input shopping-related features for a new customer
print("Enter shopping-related features for the new customer:")
new_customer_features = []
for feature_name in customer_data.columns:
    feature_value = float(input(f"Enter {feature_name}: "))
    new_customer_features.append(feature_value)

# Predict the segment for the new customer
predicted_segment = kmeans.predict([new_customer_features])[0]

# Display the predicted segment to the user
print(f"The predicted segment for the new customer is: {predicted_segment}")
