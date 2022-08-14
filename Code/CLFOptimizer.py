def SVCOPT(KernelType,x_train,x_test,y_train,y_test):
    #Optimization with PSO
    import pyswarm
    import CostFunctions
    PassingArgs=(x_train,x_test,y_train,y_test,KernelType)
    LB=[0.001,0.01]
    UB=[2,10]
    Params_opt,inaccuracy_min=pyswarm.pso(CostFunctions.SVCcost,LB,UB,args=PassingArgs)
    Accuracy_opt=1-inaccuracy_min
    return Params_opt,Accuracy_opt


def KNNOPT(x_train,x_test,y_train,y_test):
    #Optimization with PSO
    import pyswarm
    import CostFunctions
    PassingArgs=(x_train,x_test,y_train,y_test)
    LB=[2]
    UB=[50]
    Params_opt,inaccuracy_min=pyswarm.pso(CostFunctions.KNNcost,LB,UB,args=PassingArgs)
    Params_opt=int(Params_opt)
    Accuracy_opt=1-inaccuracy_min
    return Params_opt,Accuracy_opt


def MLPOPT(x_train,x_test,y_train,y_test):
    #Optimization with PSO
    import pyswarm
    import CostFunctions
    import numpy as np
    PassingArgs=(x_train,x_test,y_train,y_test)
    LB=[5,5,5]
    UB=[80,100,80]
    Params_opt,inaccuracy_min=pyswarm.pso(CostFunctions.MLPcost,LB,UB,args=PassingArgs,swarmsize=50,maxiter=30,omega=2,debug=True)
    Params_opt=np.round(Params_opt)
    Accuracy_opt=1-inaccuracy_min
    return Params_opt,Accuracy_opt


def DTOPT(x_train,x_test,y_train,y_test):
    #Optimization with PSO
    import pyswarm
    import CostFunctions
    import numpy as np
    PassingArgs=(x_train,x_test,y_train,y_test)
    LB=[10,2,1]
    UB=[10000,10,10]
    Params_opt,inaccuracy_min=pyswarm.pso(CostFunctions.DTcost,LB,UB,args=PassingArgs,debug=True,swarmsize=50,omega=0.5,maxiter=500)
    Params_opt=np.round(Params_opt)
    Accuracy_opt=1-inaccuracy_min
    return Params_opt,Accuracy_opt