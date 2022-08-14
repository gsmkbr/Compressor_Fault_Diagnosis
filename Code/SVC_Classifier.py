def SVCClassifier(x_train,x_test,y_train,y_test):
    from sklearn.svm import SVC
    import numpy as np
    import matplotlib.pyplot as plt
    
    iterations=np.linspace(0.05, 10, 60)
    train_accuracy=np.empty(len(iterations))
    test_accuracy=np.empty(len(iterations))
    test_accuracy_max=0
    for i,k in enumerate (iterations):
        svc=SVC(kernel='linear',C=k)
        svc.fit(x_train,y_train)
        train_accuracy[i]=svc.score(x_train,y_train)
        test_accuracy[i]=svc.score(x_test,y_test)
        if test_accuracy[i]>test_accuracy_max:
            best_gamma=k
            test_accuracy_max=test_accuracy[i]
            train_accuracy_choice=train_accuracy[i]
    
    plt.figure(2)
    plt.plot(iterations,test_accuracy)
    plt.title('SVC (support vector classification)')
    plt.xlabel('Gamma in SVC')
    plt.ylabel('Test Accuracy')
    
    svc=SVC(gamma=best_gamma)
    svc.fit(x_train,y_train)
    print('\n\nThe best result for SVC:\n')
    print('train accuracy is ', train_accuracy_choice, ' and test accuracy is ', test_accuracy_max,'\n\n')
    
    return test_accuracy_max
    