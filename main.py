import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Reading the dataset:

df= pd.read_csv("C:\\Users\\there\\OneDrive\\Desktop\\Minor Project\\alzheimers_disease_data.csv")
# print(df)
# print(df.head())
# print(df.tail())


# Pre-processing data:
# print(df.columns)
uw=["PatientID","DoctorInCharge"]
df= df.drop(columns=[col for col in uw if col in df.columns], errors="ignore")
# print(df)
df.fillna(df.median(numeric_only=True), inplace=True)
# print(df)
X=df.drop(columns=["Diagnosis"])
y=df["Diagnosis"]
#80:20    -->  train:test ratio  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Model Creation Setup
model=RandomForestClassifier(n_estimators=100, random_state=42)
# Model fitting (used to actually train the model)
model.fit(X_train, y_train)
print("Model training completed")



# Prediction and Reports: (this too is a part of training the model, testing with new data will be done later.)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.4f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))


# StratifiedKFold for cross validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# Perform cross-validation

cv_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
print("Cross-Validation Accuracy Scores:", cv_scores)
print(f"Mean Accuracy: {cv_scores.mean()*100:.4f}%")
print(f"Standard Deviation: {cv_scores.std()*100:.4f}%")



# Confusion Matrix:

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



