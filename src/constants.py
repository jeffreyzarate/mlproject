from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

MODELS = {
                "Logistic Regression": LogisticRegression(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "Support Vector Classifier": SVC()
            }

GRID_PARAMS = {
                "Decision Tree": {
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                "Random Forest Classifier": {
                    'n_estimators': [50, 100,],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                },
                "Support Vector Classifier": {
                    'C': [0.1, 1.0],
                    'kernel': ['linear', 'rbf'],
                },
                "Logistic Regression": {
                    'C': [0.1, 1.0],
                    'solver': ['liblinear', 'saga']
                },
                "AdaBoost Classifier": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1]
                },
                "K-Neighbors Classifier": {
                    'n_neighbors': [3, 5],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree']
                }
            }
