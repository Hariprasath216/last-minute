from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
y = iris.target  # Labels (species)

# Create a Decision Tree classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Input new flower data from the user
sepal_length = float(input("Enter sepal length (cm): "))
sepal_width = float(input("Enter sepal width (cm): "))
petal_length = float(input("Enter petal length (cm): "))
petal_width = float(input("Enter petal width (cm): "))

# Make predictions for the new flower
new_flower = [[sepal_length, sepal_width, petal_length, petal_width]]
predicted_species = clf.predict(new_flower)

# Map numeric labels to species names
species_names = iris.target_names
predicted_species_name = species_names[predicted_species[0]]

# Display the predicted species
print(f"The predicted species of the new flower is: {predicted_species_name}")
