from django.shortcuts import render,HttpResponse,redirect
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score

def view(request):
    return render(request, 'index.html')
def get_user_input(request):
# Load and preprocess the diabetes dataset
    data2 = pd.read_csv('parkinsons.csv')
# preprocess the dataset as needed
    X = data2.drop(columns=['name','status'], axis=1)
    Y = data2['status']
# Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=2)
    # Data standardization
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

# Train a  model
    clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, Y_train)


# Define a function to take user input

    if request.method == "POST":
        MDVP_Fo_Hz = float(request.POST.get('MDVP:Fo(Hz)', 0))
        MDVP_Fhi_Hz = float(request.POST.get('MDVP:Fhi(Hz)', 0))
        MDVP_Flo_Hz = float(request.POST.get('MDVP:Flo(Hz)', 0))
        MDVP_Jitter_percent = float(request.POST.get('MDVP:Jitter(%)', 0))
        MDVP_Jitter_Abs =(request.POST.get('MDVP:Jitter(Abs)', '0'))
        MDVP_RAP = (request.POST.get('MDVP:RAP', '0'))
        MDVP_PPQ = float(request.POST.get('MDVP:PPQ', 0))
        Jitter_DDP = float(request.POST.get('Jitter:DDP', 0))
        MDVP_Shimmer = float(request.POST.get('MDVP:Shimmer', 0))
        MDVP_Shimmer_dB = float(request.POST.get('MDVP:Shimmer(dB)', 0))
        Shimmer_APQ3 = float(request.POST.get('Shimmer:APQ3', 0))
        Shimmer_APQ5 = float(request.POST.get('Shimmer:APQ5', 0))
        MDVP_APQ = float(request.POST.get('MDVP:APQ', 0))
        Shimmer_DDA = float(request.POST.get('Shimmer:DDA', 0))
        NHR = float(request.POST.get('NHR', 0))
        HNR = float(request.POST.get('HNR', 0))
        RPDE = float(request.POST.get('RPDE', 0))
        DFA = float(request.POST.get('DFA', 0))
        spread1 = float(request.POST.get('spread1', 0))
        spread2 = float(request.POST.get('spread2', 0))
        D2 = float(request.POST.get('D2', 0))
        PPE = float(request.POST.get('PPE', 0))
        
 
# Call your input function to get user input
    user_input =  np.array([[MDVP_Fo_Hz,MDVP_Fhi_Hz,MDVP_Flo_Hz,MDVP_Jitter_percent,MDVP_Jitter_Abs,MDVP_RAP,MDVP_PPQ,Jitter_DDP,MDVP_Shimmer,MDVP_Shimmer_dB,Shimmer_APQ3,
                              Shimmer_APQ5,MDVP_APQ,Shimmer_DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])
    prediction = clf.predict(user_input)
    if prediction == [0]:
        # return HttpResponse("The predicted outcome : you don't have diabetes")
        return render(request, 'diapredPositive.html')
    else:
        # return HttpResponse("The predicted outcome : you have diabetes") 
        return render(request, 'diapredNegative.html')
    
    
    

    



# Create your views here.
