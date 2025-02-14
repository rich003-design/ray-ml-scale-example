**Below steps have been executed as part of the assignment:**

1. Created the repository in GitHub.
2. Cloned that repo in local.
3. Set up virtual environment using Python 3.11
4. Upgraded pip
5. Created a **requirements.txt** file
6. Created a **tune_Example.py** file to implement a ray tune example (The script l**oads the Iris dataset** and splits it into **training and test sets**.
It defines a **function train_model that trains a RandomForestClassifier** using **hyperparameters from the config dictionary.**
**Ray Tune runs multiple trials over the specified search space and reports the best hyperparameter configuration and accuracy.**)
7. Executed the example locally
8. Reviewed the output (**Ray will initialize and run several trials.
The best trial’s hyperparameters and accuracy will be printed at the end.)**

![image](https://github.com/user-attachments/assets/ba8ab6c8-24a9-4b12-85dc-f748183f5415)

9. Created a gitignore in the repository to exclude unnecessary files
10. Pushed the code to remote repository


**Commands used**
python3 -m venv venv
source venv/bin/activate
venv\Scripts\activate
pip install --upgrade pip
python tune_example.py
