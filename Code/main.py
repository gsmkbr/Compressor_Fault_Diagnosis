import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MaxExpNo=10
counter=-1
labels=['Bearing','Flywheel','Healthy','LIV','LOV','NRV','Piston','Riderbelt']
import PrimaryStatFeatures
import FFT_Module
data_columns_PrimaryStatFeatures=['Mean','Min','Max','StdDv','RMS','Skewness','Kurtosis','CrestFactor','ShapeFactor']
data_columns_Target=['Fault']
Faults={labels[0]:int(0),labels[1]:int(1),labels[2]:int(2),labels[3]:int(3),labels[4]:int(4),labels[5]:int(5),labels[6]:int(6),labels[7]:int(7)}
for label in labels:
    for ExpNo in range(1,MaxExpNo+1):
        counter+=1
        file='..\\Data\\'+label+'\\preprocess_Reading'+str(ExpNo)+'.txt'
        X=np.loadtxt(file,delimiter=',')
        if (counter%10==0): print('Lading files: ',str(counter/(len(labels)*MaxExpNo)*100),'% completed')
        StatFeatures=PrimaryStatFeatures.PrimaryFeatureExtractor(X)
        FFT_Features,data_columns_FFT_Features=FFT_Module.FFT_BasedFeatures(X)
        data_columns=data_columns_PrimaryStatFeatures+data_columns_FFT_Features+data_columns_Target
        if (label==labels[0] and ExpNo==1): data=pd.DataFrame(columns=data_columns)
        StatFeatures[0].extend(FFT_Features)
        StatFeatures[0].extend([Faults[label]])
        #print(label,ExpNo,StatFeatures)
        df_temp=pd.DataFrame(StatFeatures,index=[counter],columns=data_columns)
        data=data.append(df_temp)

input_data=data.drop(columns=['Fault'])
#normalization of input data
#reference: http://benalexkeen.com/feature-scaling-with-scikit-learn/
from sklearn import preprocessing
normalization_status='RobustScaler'   
''' Choices:
                                        1. Normalization
                                        2. StandardScaler
                                        3. MinMaxScaler
                                        4. RobustScaler
                                        5. Normalizer
                                        6. WithoutNormalization   '''
input_data_columns=data_columns_PrimaryStatFeatures+data_columns_FFT_Features

if (normalization_status=='Normalization'):
    data_array=preprocessing.normalize(input_data,norm='l2',axis=0)
    input_data=pd.DataFrame(data_array,columns=input_data_columns)
elif (normalization_status=='StandardScaler'):
    scaler = preprocessing.StandardScaler()
    scaled_df = scaler.fit_transform(input_data)
    input_data = pd.DataFrame(scaled_df, columns=input_data_columns)
elif (normalization_status=='MinMaxScaler'):
    scaler = preprocessing.MinMaxScaler()
    scaled_df = scaler.fit_transform(input_data)
    input_data = pd.DataFrame(scaled_df, columns=input_data_columns)
elif (normalization_status=='RobustScaler'):
    scaler = preprocessing.RobustScaler()
    scaled_df = scaler.fit_transform(input_data)
    input_data = pd.DataFrame(scaled_df, columns=input_data_columns)
elif (normalization_status=='Normalizer'):
    scaler = preprocessing.Normalizer()
    scaled_df = scaler.fit_transform(input_data)
    input_data = pd.DataFrame(scaled_df, columns=input_data_columns)
elif (normalization_status=='WithoutNormalization'):
    print ('No normalization is required')

target_data=pd.DataFrame(data['Fault'],columns=['Fault'],dtype=int)

DimReductionStatus=False
if (DimReductionStatus==True):
    for nComponents in range(1,110):
        #Dimensionality Reduction
        #Principal Component Analysis (PCA)
        from sklearn import decomposition
        pca = decomposition.PCA(n_components=nComponents)
        pca.fit(input_data)
        input_data_reduced = pca.transform(input_data)
        
        #Train-Test decomposition of data
        from sklearn.model_selection import train_test_split
        x_train,x_test,y_train,y_test=train_test_split(input_data_reduced,target_data,test_size=0.3,random_state=42,stratify=target_data)
    
        #Train using KNN (K NEAREST NEIGHBORS)
        import KNN_Classifier
    
        test_accuracy_max=KNN_Classifier.KNNClassifier(x_train,x_test,y_train,y_test)
        plt.figure(10)
        plt.scatter(nComponents,test_accuracy_max)
        plt.xlabel('Number of utilized components based on PCA')
        plt.ylabel('Test Accuracy')
    
        #Train using SVC(support vector classifier)
        import SVC_Classifier
        test_accuracy_max=SVC_Classifier.SVCClassifier(x_train,x_test,y_train,y_test)
        plt.figure(11)
        plt.scatter(nComponents,test_accuracy_max)
else:
    #Train-Test decomposition of data
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(input_data,target_data,test_size=0.3,random_state=42,stratify=target_data)

#Train using Decision Tree
import DT_Classifier
DT_Classifier.DTClassifier(x_train,x_test,y_train,y_test)

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#Opening the file for writing optimal classifier paramaters
file1 = open('Optimized Parameters for Classifiers.txt','w')
print('\nThis file contains the output optimized parameters for different classifiers\n\n\n',file=file1)

#Optimization of Classifier Parameters
#*************************************
SVMOptStatus=False
KNNOptStatus=False
MLPOptStatus=False
DTOptStatus=False
import CLFOptimizer

#Optimizing the SVC parameters
if (SVMOptStatus==True):
    SVM_kernels = ['linear','poly','rbf','sigmoid']
    for KernelType in SVM_kernels:
        print('\n\nstage: Optimizing SVC:',KernelType)
        SVCParams_opt,SVCAccuracy_opt=CLFOptimizer.SVCOPT(KernelType,x_train,x_test,y_train,y_test)
        print('\nClassifier: SVC-',KernelType,', Gamma=',SVCParams_opt[0],', PenaltyPrameter=',SVCParams_opt[1],', Test Acuuracy=',SVCAccuracy_opt,file=file1)

#Optimizing the KNN parameters
if (KNNOptStatus==True):
    print('\n\nstage: Optimizing KNN')
    KNNParams_opt,KNNAccuracy_opt=CLFOptimizer.KNNOPT(x_train,x_test,y_train,y_test)
    print('\nClassifier: KNN, n_neighbors=',KNNParams_opt,', Test Acuuracy=',KNNAccuracy_opt,file=file1)

#Optimizing the MLP Classifier
if (MLPOptStatus==True):
    print('\n\nstage: Optimizing MLP')
    MLPParams_opt,MLPAccuracy_opt=CLFOptimizer.MLPOPT(x_train,x_test,y_train,y_test)
    print('\nClassifier: MLP, hidden_layer_sizes=(',MLPParams_opt[0],',',MLPParams_opt[1],',',MLPParams_opt[2],'), Test Acuuracy=',MLPAccuracy_opt,file=file1)

#Optimizing the Decision Tree Classifier
if (DTOptStatus==True):
    print('\n\nstage: Optimizing Decision Tree')
    DTParams_opt,DTAccuracy_opt=CLFOptimizer.DTOPT(x_train,x_test,y_train,y_test)
    print('\nClassifier: Decision Tree, max_depth=',DTParams_opt[0],', min_samples_split=',DTParams_opt[1],', min_samples_leaf=',DTParams_opt[2],', Test Acuuracy=',DTAccuracy_opt,file=file1)

#closeing the file containing optimal classifier paramaters
file1.close()

#Generating classifiers names and their configurations
classifiers=[]
CLFnames=[]

CLFnames= CLFnames + ["SVC-linear","K-Nearest Neighbors","Multi-Layer Perceptron",
         "Decision Tree", "Random Forest", "Gaussian Process", "AdaBoost",
         "Naive Bayes", "QDA"]

#classifiers=classifiers+ [
#    SVC(kernel='linear',gamma=1.785,C=3.463),
#    KNeighborsClassifier(n_neighbors=3),
#    MLPClassifier(hidden_layer_sizes=(28,34,80,),alpha=1),
#    DecisionTreeClassifier(),
#    RandomForestClassifier(),
#    GaussianProcessClassifier(1.0 * RBF(1.0)),
#    AdaBoostClassifier(),
#    GaussianNB(),
#    QuadraticDiscriminantAnalysis()]

classifiers=classifiers+ [
    SVC(),
    KNeighborsClassifier(),
    MLPClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GaussianProcessClassifier(),
    AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=1000,learning_rate=1),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

#Writing the classification results in a file
f = open('ClassificationResults.txt','w')
print('\nThis file contains an overall comparison of different classifiers performance\n\n\n',file=f)
import ClassificationModule
ClassificationModule.Classifiers(CLFnames,classifiers,x_train,x_test,y_train,y_test)