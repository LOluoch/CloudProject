from flask import Flask, render_template, request, url_for
import psycopg2
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load dataset

data_train = pd.read_csv("loan-train.csv")
data_test = pd.read_csv("loan-test.csv")

# Define features and target variable

features = ["ApplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
target = "Loan_Status"

# Prepare data

X = data_train[features]
y = data_train[target]

imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Prepare data

(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Define PostgreSQL connection parameters
DB_HOST = "trusting_davinci"  # Name of the PostgreSQL container
DB_PORT = "5432"  # Default PostgreSQL port
DB_USER = "postgres"  # Default PostgreSQL user
DB_PASSWORD = "docker"  # Default PostgreSQL password
DB_NAME = "user_input"  # Name of the database

# Connect to PostgreSQL database
def connect_to_db():
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
    )
    return conn


# Loan Form


@app.route("/")
def loan_calc():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        input_values = [
            request.form["ApplicantIncome"],
            request.form["LoanAmount"],
            request.form["Loan_Amount_Term"],
            request.form["Credit_History"],
        ]
        input_values = [float(val) for val in input_values]

        # Insert data into PostgreSQL database
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO user_input (ApplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History) VALUES (%s, %s, %s, %s)",
            (input_values[0], input_values[1], input_values[2], input_values[3])
        )
        conn.commit()
        conn.close()

        prediction = clf.predict([input_values])[0]

        return render_template("result.html", prediction=prediction)
    else:

        return ("Method Not Allowed", 405)

                                                
