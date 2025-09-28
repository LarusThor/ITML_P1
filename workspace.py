import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import randint
from scipy.stats import loguniform
import numpy as np
import sklearn
from sklearn.feature_selection import chi2
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from ucimlrepo import fetch_ucirepo 
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import RandomizedSearchCV, cross_validate, train_test_split
from sklearn.naive_bayes import CategoricalNB


# fetch dataset 
mushroom = fetch_ucirepo(id=73) 
  
# data (as pandas dataframes) 
X = mushroom.data.features.dropna()
y = mushroom.data.targets.loc[X.index] 

# variable information 
#print(mushroom.variables) 

# sns.countplot(data=mushroom.data.features.join(mushroom.data.targets),
#               x="stalk-shape", hue="poisonous")
# plt.show()

# Important attributes:
# odor
# spore-print-color
# stalk-surface-below-ring

# sns.countplot(data=mushroom.data.features.join(mushroom.data.targets),
#               x="stalk-surface-below-ring", hue="poisonous")

# plt.show()


# # Dataset split 80/20 (A)
X_train_shroom_A, X_test_shroom_A, y_train_A, y_test_A = sklearn.model_selection.train_test_split(
    X, y, test_size=0.2, stratify=mushroom.target, random_state=40)

# # Dataset split 90/10 (B)
# X_train_shroom_B, X_test_shroom_B, y_train_B, y_test_B = sklearn.model_selection.train_test_split(
#     X, y, test_size=0.1, stratify=mushroom.target, random_state=40)

# # TODO:
# # Worth trying different methods of dealing with missing values like using most frequent value or removing instances
# # with '?' but there are circa 2000 values with missing so may be too costly

# # Comment back if dropping obsv. with missing values is not the way to go
# # imputer = SimpleImputer(missing_values='?', strategy='most_frequent')
# # imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

# # X_train_shroom_A = imputer.fit_transform(X_train_shroom_A)
# # X_test_shroom_A = imputer.fit_transform(X_test_shroom_A)

# # X_train_shroom_B = imputer.fit_transform(X_train_shroom_B)
# # X_test_shroom_B = imputer.fit_transform(X_test_shroom_B)

# #Have to create an encoder since decision tree model only works with numerical values
# encoder = OrdinalEncoder()
# X_train_shroom_A = encoder.fit_transform(X_train_shroom_A)
# X_test_shroom_A = encoder.fit_transform(X_test_shroom_A)

# chi_scores, p_values = chi2(X_train_shroom_A, y_train_A)

# chi2_results = pd.DataFrame({"Feature": X.columns,
#                              "Chi2 Score": chi_scores,
#                              "p-value": p_values}).sort_values("Chi2 Score", ascending=False)

# print(chi2_results)

# From Chi squared test
# spore-print-color
# habitat
# odor
# stalk-shape
# gill-color
# stalk-root

# model fit [0.99114064 0.99667774 1.         0.99667774 0.99778516]

selected_features = [
    "spore-print-color",
    "habitat",
    "odor",
    "stalk-shape",
    "gill-color",
    "stalk-root"
]


# selected_features = [
#     "bruises",
#     "stalk-color-above-ring",
#     "gill-spacing",
#     "stalk-color-below-ring",
#     "stalk-surface-below-ring",
#     "gill-size"
# ]



# keep only those columns in train/test
X_train_sel = X_train_shroom_A[selected_features]
X_test_sel = X_test_shroom_A[selected_features]
encoder = OrdinalEncoder()

X_train_shroom_A = encoder.fit_transform(X_train_shroom_A)
X_test_shroom_A = encoder.fit_transform(X_test_shroom_A)

X_train_sel = encoder.fit_transform(X_train_sel)
X_test_sel = encoder.fit_transform(X_test_sel)

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train_sel, y_train_A)
#training_results = cross_validate(model, X_train_sel, np.ravel(y_train_A))

#print(f"Score from categorical naive bayes model: {training_results["test_score"]}\n")

test_results = cross_validate(model, X_test_sel, np.ravel(y_test_A))
print(f"test results: {test_results["test_score"]}\n")


# # DECISION TREE EXAMPLE

# #TODO: Comment back


# # decisionTreeModel = tree.DecisionTreeClassifier()

# # decisionTreeHyperParam_dist = {
# #     "max_depth": randint(100, 500),
# #     "min_samples_split": randint(2, 30),
# #     "min_samples_leaf": randint(2, 10)
# # }

# # After testing seems like:
# # max_depth ideal range is somewhere between 250-450
# # min_samples_leaf range is between 2-5
# # min_samples_split around 8-16

# #search = RandomizedSearchCV(estimator=decisionTreeModel, param_distributions=decisionTreeHyperParam_dist, n_iter=50)

# # Seems like 80/20 and 90/10 split don't result in different outcomes so leave 80/20

# # decisionTreeHyperA = search.fit(X_train_shroom_A, y_train_A)
# # print(f"Best hyperparameters setup for 80/20 split decision tree: \n{decisionTreeHyperA.best_params_}")

# # Get all search results
# # hyperResults1 = pd.DataFrame(decisionTreeHyperA.cv_results_)

# # # Keep only the useful columns
# # hyperResults1 = hyperResults1[
# #     [
# #         "mean_test_score", 
# #         "std_test_score", 
# #         "rank_test_score", 
# #         "param_max_depth", 
# #         "param_min_samples_leaf", 
# #         "param_min_samples_split", 
# #     ]
# # ]

# # print(hyperResults1.sort_values(by="mean_test_score", ascending=False))


# # RANDOM FOREST EXAMPLE

# #TODO: Comment back
# # randomForestParams_dist = {
# #     "n_estimators": randint(100, 200),
# #     "max_depth": randint(100, 500),
# #     "min_samples_split": randint(2, 30),
# #     "min_samples_leaf": randint(2, 10),
# # }


# # randomForestModel = RandomForestClassifier()
# #randomTreeHyperSearch = RandomizedSearchCV(estimator=randomForestModel, param_distributions=randomForestParams_dist, n_iter=10)

# # randomForestHyperA = randomTreeHyperSearch.fit(X_train_shroom_A, np.ravel(y_train_A))
# # print(f"Best hyperparameters setup for 80/20 split random forest: \n{randomForestHyperA.best_params_}")

# # # Get all search results
# # hyperResults2 = pd.DataFrame(randomForestHyperA.cv_results_)

# # # Keep only the useful columns
# # hyperResults2 = hyperResults2[
# #     [
# #         "mean_test_score", 
# #         "std_test_score", 
# #         "rank_test_score", 
# #         "param_max_depth", 
# #         "param_min_samples_leaf", 
# #         "param_min_samples_split", 
# #         "param_n_estimators"
# #     ]
# # ]

# # # After testing 20 different iterations it seems that:
# # # min_samples_leaf is best in the range 2-6
# # # min_samples_split 5-25
# # # n_estimators 130-170
# # # max_depth 150-450

# # print(hyperResults2.sort_values(by="mean_test_score", ascending=False))

# # GRADIENT BOOSTING EXAMPLE

# #TODO: Comment back
# # gradientBoostingParams_dist = {
# #     "n_estimators": randint(100, 200),
# #     "learning_rate": [0.01, 0.05, 0.1, 0.2],
# #     "max_depth": randint(100, 500),
# #     "min_samples_split": randint(2, 30),
# #     "min_samples_leaf": randint(2, 10),
# #     "subsample": [0.6, 0.8, 1.0],
# #     "max_features": ["sqrt", "log2", None]
# # }

# # gradientBoostingModel = GradientBoostingClassifier()
# # gradientBoostingSearch = RandomizedSearchCV(estimator=gradientBoostingModel, param_distributions=gradientBoostingParams_dist, n_iter=10)

# # gradientBoostingHyperA = gradientBoostingSearch.fit(X_train_shroom_A, np.ravel(y_train_A))
# # print(f"Best hyperparameters setup for 80/20 split gradient boosting: \n{gradientBoostingHyperA.best_params_}")

# # # Get all search results
# # hyperResults3 = pd.DataFrame(gradientBoostingHyperA.cv_results_)

# # # Keep only the useful columns
# # hyperResults3 = hyperResults3[
# #     [
# #         "mean_test_score", 
# #         "std_test_score", 
# #         "rank_test_score", 
# #         "param_max_depth", 
# #         "param_min_samples_leaf", 
# #         "param_min_samples_split", 
# #         "param_n_estimators",
# #         "param_learning_rate",
# #         "param_subsample"
# #     ]
# # ]

# # #After running gradient boosting on 10 iterations:
# # # max_depth around 150-350
# # # min_samples_split 10-20
# # # n_estimators 110-160
# # # learning rate 0.01-0.10

# # print(hyperResults3.sort_values(by="mean_test_score", ascending=False))



# # # Naive Bayes example




# # X, y = load_iris(return_X_y=True)
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
# # gnb = GaussianNB()
# # y_pred = gnb.fit(X_train, y_train).predict(X_test)
# # print("Number of mislabeled points out of a total %d points : %d"
# #       % (X_test.shape[0], (y_test != y_pred).sum()))

# naiveBayesParams_dist = {
#     "var_smoothing": loguniform(1e-12, 1e-6)
# }

# naiveBayesModel = GaussianNB()

# naiveBayesSearch = RandomizedSearchCV(estimator=naiveBayesModel, param_distributions=naiveBayesParams_dist, n_iter=50)

# naiveBayesHyperA = naiveBayesSearch.fit(X_train_shroom_A, np.ravel(y_train_A))
# print(f"Best hyperparameters setup for 80/20 split naive bayes: \n{naiveBayesHyperA.best_params_}")

# # Get all search results
# hyperResults4 = pd.DataFrame(naiveBayesHyperA.cv_results_)

# # Keep only the useful columns
# hyperResults4 = hyperResults4[
#     [
#         "mean_test_score", 
#         "std_test_score", 
#         "rank_test_score", 
#         "param_var_smoothing"
#     ]
# ]

# # After testing 50 iterations we can see:
# # best range for var_smoothing is between 8.46×10−12 to 8.63×10−7

# print(hyperResults4.sort_values(by="mean_test_score", ascending=False))