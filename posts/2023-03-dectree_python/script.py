import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.tree import export_graphviz
import graphviz
import seaborn as sns 
from palmerpenguins import load_penguins
from sklearn.preprocessing import StandardScaler

# Load the dataset
#pip install palmerpenguins



sns.set_style('whitegrid')
penguins = load_penguins()
penguins.head()
penguins = penguins.dropna()
# Check for missing values
print(penguins.isnull().sum())

# Plot the distribution of the target variable
plt.hist(penguins['species'])
plt.show()

# Plot the pairwise relationships between the features
pd.plotting.scatter_matrix(penguins, diagonal='hist')
plt.show()

sns.pairplot(data=penguins, hue='species')
plt.show()
# Make a copy of data_size DataFrame
data_df = penguins.copy()

# Define categorical variables
cat_vars = ['island', 'sex']

# Iterate through categorical variables and create dummy variables
dummy_vars = []
for var in cat_vars:
    dummy_var = pd.get_dummies(data_df[var], prefix=var)
    dummy_vars.append(dummy_var)
    
dummy_df = pd.concat(dummy_vars, axis=1)
data_df = pd.concat([data_df, dummy_df], axis=1)
data_df = data_df.drop(cat_vars, axis=1)

data_final_vars = ['bill_length_mm','bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'island_Biscoe',
 'island_Dream', 'sex_female']
y = penguins['species'].values
X = data_df[data_final_vars].values

#escala
scaler = StandardScaler()
# transform data
scaled = scaler.fit_transform(penguins[['bill_length_mm','bill_depth_mm', 'flipper_length_mm', 'body_mass_g']])
scaled_df = pd.DataFrame(scaled, columns = ['bill_length_mm','bill_depth_mm', 'flipper_length_mm', 'body_mass_g'])
scaled_df.head()

# copying the scaled data back to the main dataframe
data_df[['bill_length_mm','bill_depth_mm', 'flipper_length_mm', 'body_mass_g']] = scaled_df[['bill_length_mm','bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
penguins = data_df



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the decision tree model
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# Make predictions on the test set
y_pred = tree.predict(X_test)

# Evaluate the model
print("Accuracy: {:.2f}%".format(100 * accuracy_score(y_test, y_pred)))
print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_pred, average='weighted')))
print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_pred, average='weighted')))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

score = f1_score(y_test, y_pred, average='macro')
score


# Visualize the decision tree
dot_data = export_graphviz(tree, out_file=None, 
                           feature_names=data_final_vars,  
                           class_names=['Adelie', 'Chinstrap', 'Gentoo'],  
                           filled=True, rounded=True,  
                           special_characters=True)  

graph = graphviz.Source(dot_data)  
graph.render("penguins_decision_tree.png")  
















