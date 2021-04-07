def get_details():
    print("Loan status Prediction for user")
    print("Enter the required details to check if you're eligible for a loan approval.\n")
    flag = False
    while(flag == False):
        try:
            gender = int(input("Gender: Enter 1 for Male and 2 for Female. "))
            if gender != 1 and gender != 2:
                print("Wrong input. Try again!")
                continue
            flag = True
        except:
            print("Wrong input. Try again!")
            continue
    flag = False
    while(flag == False):
        try:
            married = int(
                input("Are you Married?: Enter 1 for Yes and 2 for No. "))
            if married != 1 and married != 2:
                print("Wrong input. Try again!")
                continue
            flag = True
        except:
            print("Wrong input. Try again!")
            continue
    flag = False
    while(flag == False):
        try:
            dependents = int(input("Enter number of dependents: "))
            if dependents < 0:
                print("Wrong input. Try again!")
                continue
            if dependents > 2:
                dependents = 3
            flag = True
        except:
            print("Wrong input. Try again!")
            continue
    flag = False
    while(flag == False):
        try:
            education = int(
                input("Education: Enter 1 for Graduate and 2 for Not Graduate"))
            if education != 1 and education != 2:
                print("Wrong input. Try again!")
                continue
            flag = True
        except:
            print("Wrong input. Try again!")
            continue
    flag = False
    while(flag == False):
        try:
            selfemployed = int(
                input("Are you Self-Employed?: Enter 1 for Yes and 2 for No. "))
            if selfemployed != 1 and selfemployed != 2:
                print("Wrong input. Try again!")
                continue
            flag = True
        except:
            print("Wrong input. Try again!")
            continue
    flag = False
    while(flag == False):
        try:
            income = int(input("Enter your Income: "))
            flag = True
        except:
            print("Wrong input. Try again!")
            continue
    flag = False
    while(flag == False):
        try:
            coincome = int(
                input("Enter your Co-applicants Income: (Enter 0 is there is no coapplicant) "))
            flag = True
        except:
            print("Wrong input. Try again!")
            continue
    flag = False
    while(flag == False):
        try:
            loan = int(input("Enter the loan amount required: "))
            flag = True
        except:
            print("Wrong input. Try again!")
            continue
    flag = False
    while(flag == False):
        try:
            loanterm = int(input("Enter the loan term required: "))
            flag = True
        except:
            print("Wrong input. Try again!")
            continue
    flag = False
    while(flag == False):
        try:
            credit_history = int(
                input("Do you have a credit history?: Enter 1 for Yes and 2 for No. "))
            if credit_history != 1 and credit_history != 2:
                print("Wrong input. Try again!")
                continue
            flag = True
        except:
            print("Wrong input. Try again!")
            continue
    flag = False
    while(flag == False):
        try:
            area = int(
                input("Enter you area: 1 for Rural, 2 for Semiurban and 3 for Urban: "))
            if area != 1 and area != 2 and area != 3:
                print("Wrong input. Try again!")
                continue
            flag = True
        except:
            print("Wrong input. Try again!")
            continue
    dtt = {
        "Gender": [gender],
        "Married": [married],
        "Education": [education],
        "Self_Employed": [selfemployed],
        "ApplicantIncome": [income],
        "CoapplicantIncome": [coincome],
        "LoanAmount": [loan],
        "Loan_Amount_Term": [loanterm],
        "Credit_History": [credit_history],
        "Property_Area": [area],
        "Dependents": [dependents]
    }
    return dtt   # Returns a dictionary with structure of dataframe required for prediction
