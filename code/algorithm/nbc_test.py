import pandas as pd
from nbc_classifier import NBC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split




#testowanie dla pliku glass.csv

glassDf = pd.read_csv('glass.csv')
train_df, test_df = train_test_split(glassDf, test_size=0.2, random_state=42)
className = 'Type'

X_train = train_df.loc[:,train_df.columns!=className]
y_train = train_df.loc[:,train_df.columns==className]
X_test = test_df.loc[:,test_df.columns!=className]
y_test = test_df.loc[:,test_df.columns==className]

#testowanie dla gotowego zbioriu iris

# iris = load_iris()
# X = iris.data  # Features
# y = iris.target  # Labels
# className = 'Species'
#
# iris_df = pd.DataFrame(data=X, columns=iris.feature_names)
# iris_df[className] = y
# train_df, test_df = train_test_split(iris_df, test_size=0.2, random_state=42)
#
# X_train = train_df.loc[:,train_df.columns!=className]
# y_train = train_df.loc[:,train_df.columns==className]
# X_test = test_df.loc[:,test_df.columns!=className]
# y_test = test_df.loc[:,test_df.columns==className]

nbc = NBC(0.6)

nbc.fit(X_train, y_train)

print(nbc.score(X_test, y_test))

