import pandas as pd
from sklearn.ensemble import RandomForestClassifier

"""https://www.kaggle.com/competitions/loan-approval-quantitativedata
This a Kaggle competition in order to predict the risk of default of credit
I obtain a score of 0,76 for this model.
I used a random forest method to obtain the probabiliy of risk defaut
I load the data and after define X_train, X_test and Y_train as Y_test was not
given
"""


train = pd.read_csv(r"C:\Users\Mathis\Documents\Codage\train.csv")
test = pd.read_csv(r"C:\Users\Mathis\Documents\Codage\test.csv")


train_clean = train.dropna()
test_clean = test.dropna()

# We convert the float features into numerical one
for col in ['person_home_ownership', 'loan_intent', 'cb_person_default_on_file', 'loan_grade']:
    train_clean[col] = pd.factorize(train_clean[col])[0]
    test_clean[col] = pd.factorize(test_clean[col])[0]

# We define X_train, X_test,Y_train
Y_train = train_clean['loan_status']
X_train = train_clean.drop(columns=['loan_status'])
X_test=test_clean

# Ensure that the columns in X_test match those in X_train
X_test = X_test[X_train.columns]

#We apply the random forest model
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)

probabilities = rf.predict_proba(X_test)

result = pd.DataFrame({
    'id': test_clean['id'],  # Identifiant des échantillons
    'loan_status': probabilities[:, 1]  # Probabilités pour la classe 1 (1er indice)
})
result.to_csv(r"C:\Users\Mathis\Documents\Codage\submission.csv", index=False)
