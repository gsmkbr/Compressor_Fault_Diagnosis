def DTClassifier(x_train,x_test,y_train,y_test):
    from sklearn.tree import DecisionTreeClassifier
    import numpy as np
    import matplotlib.pyplot as plt
    
    dtc=DecisionTreeClassifier()
    dtc.fit(x_train,y_train)
    train_accuracy_dt=dtc.score(x_train,y_train)
    test_accuracy_dt=dtc.score(x_test,y_test)
    print('\n\nThe best result for decision tree:\n')
    print('train accuracy is ', train_accuracy_dt, ' and test accuracy is ', test_accuracy_dt,'\n\n')

    
#    iterations=np.linspace(0.05, 3, 60)
#    train_accuracy=np.empty(len(iterations))
#    test_accuracy=np.empty(len(iterations))
#    test_accuracy_max=0
#    for i,k in enumerate (iterations):
#        svc=SVC(gamma=k)
#        svc.fit(x_train,y_train)
#        train_accuracy[i]=svc.score(x_train,y_train)
#        test_accuracy[i]=svc.score(x_test,y_test)
#        if test_accuracy[i]>test_accuracy_max:
#            best_gamma=k
#            test_accuracy_max=test_accuracy[i]
#            train_accuracy_choice=train_accuracy[i]
#    
#    plt.figure(2)
#    plt.plot(iterations,test_accuracy)
#    plt.title('SVC (support vector classification)')
#    plt.xlabel('Gamma in SVC')
#    plt.ylabel('Test Accuracy')
#    
#    svc=SVC(gamma=best_gamma)
#    svc.fit(x_train,y_train)
#    print('\n\nThe best result for SVC:\n')
#    print('train accuracy is ', train_accuracy_choice, ' and test accuracy is ', test_accuracy_max,'\n\n')