from django.shortcuts import render, HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel,InputData
from django.conf import settings
import pandas as pd


# # Create your views here.
# def UserRegisterActions(request):
#     if request.method == 'POST':
#         form = UserRegistrationForm(request.POST)
#         if form.is_valid():
#             print('Data is Valid')
#             form.save()
#             messages.success(request, 'You have been successfully registered')
#             form = UserRegistrationForm()
#             return render(request, 'UserRegistrations.html', {'form': form})
#         else:
#             messages.success(request, 'Email or Mobile Already Existed')
#             print("Invalid form")
#     else:
#         form = UserRegistrationForm()
#     return render(request, 'UserRegistrations.html', {'form': form})


# def UserLoginCheck(request):
#     if request.method == "POST":
#         loginid = request.POST.get('loginname')
#         pswd = request.POST.get('pswd')
#         print("Login ID = ", loginid, ' Password = ', pswd)
#         try:
#             check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
#             status = check.status
#             print('Status is = ', status)
#             if status == "activated":
#                 request.session['id'] = check.id
#                 request.session['loggeduser'] = check.name
#                 request.session['loginid'] = loginid
#                 request.session['email'] = check.email
#                 print("User id At", check.id, status)
#                 return render(request, 'users/UserHome.html', {})
#             else:
#                 messages.success(request, 'Your Account Not at activated')
#                 return render(request, 'UserLogin.html')
#         except Exception as e:
#             print('Exception is ', str(e))
#             pass
#         messages.success(request, 'Invalid Login id and password')
#     return render(request, 'UserLogin.html', {})


# def UserHome(request):
#     return render(request, 'users/UserHome.html', {})


# def UserViewData(request):
#     path = settings.MEDIA_ROOT + "\\" + "dataset.csv"
#     df = pd.read_csv(path, nrows=100)
#     df = df.to_html(index=False)
#     return render(request, 'users/UserViewDataset.html', {'data': df})


# def PreprocessedData(request):
#     path = settings.MEDIA_ROOT + "\\" + "dataset.csv"
#     df = pd.read_csv(path)
#     df['Genre'].replace({'Female': 0, 'Male': 1}, inplace=True)
#     df['TypeEtab'].replace({'Public': 0, 'Private': 1}, inplace=True)
#     df['Niveau'].replace({'Primary': 1, 'Secondary': 2, 'Tertiary': 3}, inplace=True)
#     df['RetardSco'].replace({'1 year': 1, '2 years': 2, 'None': 0}, inplace=True)
#     df['Provenance'].replace({'Rural': 1, 'Suburban': 2, 'Urban': 3}, inplace=True)
#     df['Fee_reimbursement'].replace({'Yes': 1, 'No': 0}, inplace=True)
#     df['Result'].replace({'Continue': 0, 'Discontinue': 1}, inplace=True)
#     df = df.to_html(index=False)
#     return render(request, 'users/UserViewDataset.html', {'data': df})


# def MLResults(request):
#     from .algorithms import ImplementAlgorithmsCodes
#     knn = ImplementAlgorithmsCodes.knnResults()
#     rf = ImplementAlgorithmsCodes.randomForest()
#     svm = ImplementAlgorithmsCodes.svmAlgorithm()
#     # sgd = ImplementAlgorithmsCodes.sgdAlgorithm()
#     return render(request, 'users/mlresultst.html', {'knn': knn, 'rf': rf, 'svm': svm})


# def heatMapDraw(request):
#     from .algorithms import ImplementAlgorithmsCodes
#     ImplementAlgorithmsCodes.corrGraph()
#     ImplementAlgorithmsCodes.randomForest()
#     return render(request, 'users/corrGraph.html', {'rcm':rcm})


# def predict_result(request):
#     import pandas as pd
#     import numpy as np
#     from django.shortcuts import render
#     from sklearn.model_selection import train_test_split
#     from sklearn.neighbors import KNeighborsClassifier
#     from django.conf import settings
#     from sklearn.metrics import accuracy_score, precision_score, recall_score
    
#     if request.method == "POST":
#         try:
#             # Get input values from form
#             TypeEtab = int(request.POST['TypeEtab'])
#             Genre = int(request.POST['Genre'])
#             Niveau = int(request.POST['Niveau'])
#             RetardSco = int(request.POST['RetardSco'])
#             Provenance = int(request.POST['Provenance'])
#             Fee_reimbursement = int(request.POST['Fee_reimbursement'])
#             Moy = float(request.POST['Moy'])

#             # Convert input to NumPy array and reshape
#             input_data = np.array([[TypeEtab, Genre, Niveau, RetardSco, Provenance, Fee_reimbursement, Moy]])
#             path1 = settings.MEDIA_ROOT + '//' + 'dataset.csv'
#             df = pd.read_csv(path1)
#             df = df.dropna()

#             # Encoding categorical variables
#             df['Genre'].replace({'Female': 0, 'Male': 1}, inplace=True)
#             df['TypeEtab'].replace({'Public': 0, 'Private': 1}, inplace=True)
#             df['Niveau'].replace({'Primary': 1, 'Secondary': 2, 'Tertiary': 3}, inplace=True)
#             df['RetardSco'].replace({'1 year': 1, '2 years': 2, 'None': 0}, inplace=True)
#             df['Provenance'].replace({'Rural': 1, 'Suburban': 2, 'Urban': 3}, inplace=True)
#             df['Handicap'].replace({'Yes': 1, 'No': 0}, inplace=True)
#             df['Fee_reimbursement'].replace({'Yes': 1, 'No': 0}, inplace=True)
#             df['Result'].replace({'Continue': 0, 'Discontinue': 1}, inplace=True)

#             # Splitting dataset
#             X = df[['Genre','TypeEtab','Niveau','RetardSco','Provenance','Fee_reimbursement','Moy']]
#             y = df['Result']

#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#             # Train KNN Model
#             knn = KNeighborsClassifier()
#             knn.fit(X_train, y_train)

#             # Calculate model performance
#             y_pred = knn.predict(X_test)
#             accuracy = accuracy_score(y_test, y_pred)
#             precision = precision_score(y_test, y_pred,pos_label=1)
#             recall = recall_score(y_test, y_pred,pos_label=1)
            
#             print(f"Accuracy: {accuracy}")
#             print(f"Precision: {precision}")
#             print(f"Recall: {recall}")

#             # Predict result
#             prediction = knn.predict(input_data)[0]
#             print('The prediction is:', prediction)

#             # Map the prediction to the result
#             result = "Continue" if prediction == 0 else "Discontinue"
#             print('The result is:', result)

#             return render(request, 'users/predict.html', {'result': result})

#         except Exception as e:
#             print("Exception occurred:", str(e))
#             return render(request, 'users/predict.html', {'error': str(e)})
    
#     else:
#         return render(request, 'users/predict.html')


from django.shortcuts import render
from django.contrib import messages
from django.conf import settings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sb
import matplotlib.pyplot as plt


def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.error(request, 'Email or Mobile Already Exists')
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                return render(request, 'users/UserHome.html')
            else:
                messages.error(request, 'Your Account is Not Activated')
        except Exception as e:
            messages.error(request, 'Invalid Login ID or Password')
    return render(request, 'UserLogin.html')


def UserHome(request):
    return render(request, 'users/UserHome.html')


def UserViewData(request):
    path = settings.MEDIA_ROOT + "\\" + "dataset.csv"
    df = pd.read_csv(path, nrows=100)
    df = df.to_html(index=False)
    user_data = InputData.objects.all()[:5]
    print(user_data)
    return render(request, 'users/UserViewDataset.html', {'data': df,'user_data':user_data})


def PreprocessedData(request):
    df = load_and_preprocess_data()
    df = df.to_html(index=False)
    return render(request, 'users/UserViewDataset.html', {'data': df})


from django.shortcuts import render, HttpResponse
from django.contrib import messages
from django.conf import settings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sb
import matplotlib.pyplot as plt

# Helper function to load and preprocess the data
def load_and_preprocess_data():
    """Load and preprocess dataset."""
    path = settings.MEDIA_ROOT + "\\" + "dataset.csv"
    df = pd.read_csv(path)
    df = df.dropna()  # Drop rows with missing values
    print(df)
    # Encoding categorical variables to numeric
    df['Genre'].replace({'Female': 0, 'Male': 1}, inplace=True)
    df['TypeEtab'].replace({'Public': 0, 'Private': 1}, inplace=True)
    df['Niveau'].replace({'Primary': 1, 'Secondary': 2, 'Tertiary': 3}, inplace=True)
    df['RetardSco'].replace({'1 year': 1, '2 years': 2, 'None': 0}, inplace=True)
    df['Provenance'].replace({'Rural': 1, 'Suburban': 2, 'Urban': 3}, inplace=True)
    df['Fee_reimbursement'].replace({'Yes': 1, 'No': 0}, inplace=True)
    df['Result'].replace({'Continue': 0, 'Discontinue': 1}, inplace=True)

    return df

# Refactored MLResults function for model evaluation
def MLResults(request):
    # Load and preprocess the dataset
    df = load_and_preprocess_data()

    # Define features and target
    X = df[['Genre', 'TypeEtab', 'Niveau', 'RetardSco', 'Provenance', 'Fee_reimbursement', 'Moy']]
    y = df['Result']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize classifiers
    knn = KNeighborsClassifier()
    rf = RandomForestClassifier()
    svm = SVC()

    # Train KNN model
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    knn_acc = accuracy_score(y_test, knn_pred)
    knn_precision = precision_score(y_test, knn_pred)
    knn_recall = recall_score(y_test, knn_pred)
    knn_cm = confusion_matrix(y_test, knn_pred)

    plt.figure(figsize=(6, 5))
    sb.heatmap(knn_cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.title('KNN Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    # Train Random Forest model
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_precision = precision_score(y_test, rf_pred)
    rf_recall = recall_score(y_test, rf_pred)
    rf_cm = confusion_matrix(y_test, rf_pred)

    plt.figure(figsize=(6, 5))
    sb.heatmap(rf_cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.title('Random Forest Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    print(rf_cm)


    # Train SVM model
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_precision = precision_score(y_test, svm_pred)
    svm_recall = recall_score(y_test, svm_pred)
    svm_cm = confusion_matrix(y_test, svm_pred)


    plt.figure(figsize=(6, 5))
    sb.heatmap(svm_cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.title('SVM Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()


    # Return the evaluation results to the template
    return render(request, 'users/mlresultst.html', {
        'knn_acc': knn_acc,
        'knn_precision': knn_precision,
        'knn_recall': knn_recall,
        'rf_acc': rf_acc,
        'rf_precision': rf_precision,
        'rf_recall': rf_recall,
        'rf_cm': rf_cm,
        'svm_acc': svm_acc,
        'svm_precision': svm_precision,
        'svm_recall': svm_recall
    })
import os
# Function to generate and display the correlation heatmap
def corrGraph():
    """Generate and display the correlation heatmap."""
    print("corr is called")
    path = settings.MEDIA_ROOT + "\\" + "dataset.csv"
    df = pd.read_csv(path)
    df['Genre'].replace({'Female': 0, 'Male': 1}, inplace=True)
    df['TypeEtab'].replace({'Public': 0, 'Private': 1}, inplace=True)
    df['Niveau'].replace({'Primary': 1, 'Secondary': 2, 'Tertiary': 3}, inplace=True)
    df['RetardSco'].replace({'1 year': 1, '2 years': 2, 'None': 0}, inplace=True)
    df['Provenance'].replace({'Rural': 1, 'Suburban': 2, 'Urban': 3}, inplace=True)
    df['Fee_reimbursement'].replace({'Yes': 1, 'No': 0}, inplace=True)
    df['Result'].replace({'Continue': 0, 'Discontinue': 1}, inplace=True)

    selected_columns = ['Genre', 'TypeEtab', 'Niveau', 'RetardSco', 'Provenance', 'Fee_reimbursement', 'Moy']
    selected_df = df[selected_columns]
    corr = selected_df.corr()
    print(corr)
    # Generate heatmap
    plt.figure(figsize=(6, 6))
    sb.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Heatmap")
    save_dir = os.path.join(settings.BASE_DIR, 'assets','static', 'images')

    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # Create the directory if it doesn't exist

    save_path = os.path.join(save_dir, 'corr_graph.png')
    plt.savefig(save_path)  
    print("Heatmap has been saved at:", save_path)  
    plt.close()

# View to render heatmap page
def heatMapDraw(request):

    corrGraph()  # Generate the heatmap    
    return render(request, 'users/corrGraph.html')

# Function to handle user prediction using the trained KNN model
from .models import InputData
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from django.shortcuts import render

def predict_result(request):
    if request.method == "POST":
        try:
            # Extract input data from the form
            print("Extracting input data from the form...")
            input_data = get_input_data(request)
            print(f"Input data extracted: {input_data}")

            # Load and preprocess data (Assuming you have a function for this)
            print("Loading and preprocessing data...")
            df = load_and_preprocess_data()
            print(f"Data loaded: {df.head()}")

            # Split the dataset for model training
            X = df[['Genre', 'TypeEtab', 'Niveau', 'RetardSco', 'Provenance', 'Fee_reimbursement', 'Moy']]
            y = df['Result']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(f"Data split: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

            # Train KNN model
            print("Training KNN model...")
            knn = KNeighborsClassifier()
            knn.fit(X_train, y_train)
            print("Model trained.")

            # Predict result based on input data
            print("Making prediction...")
            prediction = knn.predict(input_data)[0]
            result = "Continue" if prediction == 0 else "Discontinue"
            print(f"Prediction made: {result}")

            # Save the input data to the database
            input_instance = InputData(
                TypeEtab=request.POST['TypeEtab'],
                Genre=request.POST['Genre'],
                Niveau=request.POST['Niveau'],
                RetardSco=request.POST['RetardSco'],
                Provenance=request.POST['Provenance'],
                Fee_reimbursement=request.POST['Fee_reimbursement'],
                Moy=request.POST['Moy'],
                Result=result  # Use the predicted result
            )
            input_instance.save()
            print(f"Input instance saved: {input_instance}")

            return render(request, 'users/predict.html', {'result': result})

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return render(request, 'users/predict.html', {'error': str(e)})

    else:
        return render(request, 'users/predict.html')

# Helper function to extract input data from the form
def get_input_data(request):
    """Extract input data from the POST request."""
    print("Extracting form data...")
    TypeEtab = int(request.POST['TypeEtab'])
    Genre = int(request.POST['Genre'])
    Niveau = int(request.POST['Niveau'])
    RetardSco = int(request.POST['RetardSco'])
    Provenance = int(request.POST['Provenance'])
    Fee_reimbursement = int(request.POST['Fee_reimbursement'])
    Moy = float(request.POST['Moy'])

    print(f"Extracted data: TypeEtab={TypeEtab}, Genre={Genre}, Niveau={Niveau}, RetardSco={RetardSco}, Provenance={Provenance}, Fee_reimbursement={Fee_reimbursement}, Moy={Moy}")
    
    return np.array([[TypeEtab, Genre, Niveau, RetardSco, Provenance, Fee_reimbursement, Moy]])



