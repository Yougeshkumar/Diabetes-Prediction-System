# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
import warnings
import joblib
warnings.filterwarnings("ignore")

# Load dataset
data = pd.read_csv(r"E:\Engineering\5Sem\MiniProject\Datasets\diabetes.csv")
print("Initial Data Shape:", data.shape)

# Handle Missing Values
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[columns_with_zeros] = data[columns_with_zeros].replace(0, np.nan)
data.fillna(data.median(), inplace=True)

# Split features and target
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# Outlier Removal using Isolation Forest
iso_forest = IsolationForest(contamination=0.02, random_state=42)
outliers = iso_forest.fit_predict(X)
X = X[outliers == 1]
y = y[outliers == 1]
print("Data Shape after Outlier Removal:", X.shape)

# Feature Engineering with Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# Address Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)
print("Data Shape after SMOTE:", X_res.shape)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# Hyperparameter Optimization for LightGBM
lgbm_clf = LGBMClassifier(random_state=42)
lgbm_params = {'n_estimators': [200, 300, 500], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [10, 15, -1]}
lgbm_grid = RandomizedSearchCV(lgbm_clf, lgbm_params, cv=5, n_iter=10, random_state=42, n_jobs=-1)
lgbm_grid.fit(X_train, y_train)
print("Best LGBM Params:", lgbm_grid.best_params_)
lgbm_best = lgbm_grid.best_estimator_

# Base Classifiers
rf_clf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
gb_clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
svc_clf = SVC(probability=True, C=1, kernel='rbf', random_state=42)
knn_clf = KNeighborsClassifier(n_neighbors=7, weights='distance')

# Stacking Classifier
estimators = [
    ('rf', rf_clf),
    ('gb', gb_clf),
    ('svc', svc_clf),
    ('knn', knn_clf)
]
stack_clf = StackingClassifier(estimators=estimators, final_estimator=lgbm_best)

# Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('stack', stack_clf),
    ('lgbm', lgbm_best)
], voting='soft')

# Train the Voting Classifier
voting_clf.fit(X_train, y_train)

# Make Predictions
y_pred = voting_clf.predict(X_test)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
roc_auc = roc_auc_score(y_test, voting_clf.predict_proba(X_test)[:, 1])
print("ROC-AUC:", roc_auc)

# Save the model and scaler
joblib.dump(voting_clf, 'final_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(poly, 'poly_transformer.pkl')

print("Model training complete and saved!")
