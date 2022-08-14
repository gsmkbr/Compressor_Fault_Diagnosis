def Classifiers(names,classifiers,x_train,x_test,y_train,y_test):
    import matplotlib.pyplot as plt
    counter=0
    plt.figure(3)
    BestPredictionScore=0
    for name, clf in zip(names, classifiers):
        counter+=1
        clf.fit(x_train, y_train)
        TrainingScore = clf.score(x_train,y_train)
        PredictionScore = clf.score(x_test, y_test)
        f = open('ClassificationResults.txt','a')
        print('Training accuracy for ',name,' is: ',TrainingScore,' and Prediction accuracy is: ',PredictionScore,file=f)
        f.close()
        if (PredictionScore>BestPredictionScore): 
            BestPredictionScore=PredictionScore
            BestConfig=clf
        #plt.plot(counter,PredictionScore)
#        print('\nTraining accuracy for ',name,' is: ',TrainingScore,' and Prediction accuracy is: ',PredictionScore)
    print('\n\nBestPredictionScore=',str(BestPredictionScore),'\nBest configureation:\n',BestConfig)