from flask import Flask, render_template, request, url_for
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load dataset
data_train = pd.read_csv('loan-train.csv')
data_test = pd.read_csv('loan-test.csv')

# Define features and target variable
features = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
target = 'Loan_Status'

# Prepare data
X = data_train[features]
y = data_train[target]

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

@app.route("/")
def loan_calc():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_values = [request.form['ApplicantIncome'], request.form['LoanAmount'],
                        request.form['Loan_Amount_Term'], request.form['Credit_History']]
        input_values = [float(val) for val in input_values]
        
        prediction = clf.predict([input_values])[0]
        
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
