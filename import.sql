create schema csvs;
create table csvs.loan_train(
    ApplicantIncome text,
    LoanAmount text,
    Loan_Amount_Term text,
    Credit_History text,
    Loan_Status text,
    description text
);
copy csvs.loan_train
from '/home/ec2-user/data'
delimiter ',' header csv;