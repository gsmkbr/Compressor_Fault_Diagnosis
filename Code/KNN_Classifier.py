def KNNClassifier(x_train,x_test,y_train,y_test):
    from sklearn.neighbors import KNeighborsClassifier
    import matplotlib.pyplot as plt
    import numpy as np
    iterations=np.arange(3,50)
    train_accuracy=np.empty(len(iterations))
    test_accuracy=np.empty(len(iterations))
    test_accuracy_max=0
    for i,k in enumerate (iterations):
        knn_model=KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(x_train,y_train)
        train_accuracy[i]=knn_model.score(x_train,y_train)
        test_accuracy[i]=knn_model.score(x_test,y_test)
        if test_accuracy[i]>test_accuracy_max:
            best_neighbors=k
            test_accuracy_max=test_accuracy[i]
            train_accuracy_choice=train_accuracy[i]
        #print('for ',k,'neighbors, train accuracy is ', train_accuracy[i], ' and test accuracy is ', test_accuracy[i])
    plt.figure(1)
    plt.title('KNN CLASSIFICATION')
    plt.plot(iterations,test_accuracy)
    plt.xlabel('number of neighbors')
    plt.ylabel('Test Accuracy')
    
    
    #printing results for best solution
    knn_model=KNeighborsClassifier(n_neighbors=best_neighbors)
    knn_model.fit(x_train,y_train)
    print('\n\nThe best result for KNN Classifier:\n')
    print('train accuracy is ', train_accuracy_choice, ' and test accuracy is ', test_accuracy_max,'\n\n')
    
    print('\n\n\nKNN Predicted Faults VS Target Faults:\n\n')
    PredictedFaults=np.array(knn_model.predict(x_test))
    TargetFaults=np.array(y_test).reshape(len(PredictedFaults),)
    for i in range (0,len(PredictedFaults)):
        print('Target Fault: ',TargetFaults[i], '  Predicted Fault: ',PredictedFaults[i])
    
    return test_accuracy_max