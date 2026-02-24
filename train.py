# model training
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

#Load data
X, y = load_iris(return_X_y=True)
#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2)

#Start MLflow experiment
with mlflow.start_run():
    #Train model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    #Predict and evaluate
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    #Log parameters and metrics
    mlflow.log_param("max_iter", 200)
    mlflow.log_metric("accuracy", acc)
    
    #Save model
    joblib.dump(model, "models.pkl")
    print("Model trained and saved")