# tune_example.py
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import ray
from ray import tune

def train_model(config):
    # Load the Iris dataset
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42
    )
    
    # Create a RandomForestClassifier with hyperparameters from config
    clf = RandomForestClassifier(
        n_estimators=int(config["n_estimators"]),
        max_depth=int(config["max_depth"])
    )
    
    # Train the model
    clf.fit(X_train, y_train)
    
    # Evaluate the model
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    # Report the accuracy back to Ray Tune
    tune.report(accuracy=acc)

if __name__ == "__main__":
    # Initialize Ray
    ray.init()
    
    # Define a search space for hyperparameters
    config = {
        "n_estimators": tune.choice([10, 50, 100, 200]),
        "max_depth": tune.choice([2, 4, 6, 8, 10])
    }
    
    # Run hyperparameter tuning using Ray Tune
    analysis = tune.run(
        train_model,
        config=config,
        metric="accuracy",
        mode="max",
        num_samples=10,              # Number of trials
        resources_per_trial={"cpu": 1}
    )
    
    # Get the best trial results
    best_trial = analysis.get_best_trial("accuracy", "max", "last")
    print("Best trial config: ", best_trial.config)
    print("Best trial final accuracy: ", best_trial.last_result["accuracy"])
