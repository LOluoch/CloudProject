from flask import Flask, render_template, request, url_for
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load dataset
data_train = pd.read_csv('loan_train.csv')
data_test = pd.read_csv('loan_test.csv')

# Define features and target variable
features = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
target = 'Loan_Status'

# Prepare data
X_train = data_train[features]
y_train = data_train[target]
X_test = data_test[features]
y_test = data_test[target]

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

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
