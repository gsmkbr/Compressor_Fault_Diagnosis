def SVCcost(Params,*args):
    from sklearn.svm import SVC
    x_train,x_test,y_train,y_test,KernelType=args
    svc=SVC(kernel=KernelType,gamma=Params[0],C=Params[1])
    svc.fit(x_train,y_train)
    test_accuracy=svc.score(x_test,y_test)
    svc_cost=1-test_accuracy
    return svc_cost


def KNNcost(Params,*args):
    from sklearn.neighbors import KNeighborsClassifier
    x_train,x_test,y_train,y_test=args
    knn=KNeighborsClassifier(n_neighbors=int(Params[0]))
    knn.fit(x_train,y_train)
    test_accuracy=knn.score(x_test,y_test)
    knn_cost=1-test_accuracy
    return knn_cost


def MLPcost(Params,*args):
    from sklearn.neural_network import MLPClassifier
    x_train,x_test,y_train,y_test=args
    mlp=MLPClassifier(hidden_layer_sizes=(int(Params[0]),int(Params[1]),int(Params[2]),),alpha=1)
    mlp.fit(x_train,y_train)
    test_accuracy=mlp.score(x_test,y_test)
    mlp_cost=1-test_accuracy
    return mlp_cost


def DTcost(Params,*args):
    from sklearn.tree import DecisionTreeClassifier
    x_train,x_test,y_train,y_test=args
    dt=DecisionTreeClassifier(max_depth=int(Params[0]),min_samples_split=int(Params[1]),min_samples_leaf=int(Params[2]))
    dt.fit(x_train,y_train)
    test_accuracy=dt.score(x_test,y_test)
    dt_cost=1-test_accuracy
    return dt_cost