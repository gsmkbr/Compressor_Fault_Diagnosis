def FFT_BasedFeatures(X):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.fftpack import fft
    NBins=100
    N=len(X)
    T=1/N
    FreqList=np.linspace(0.0, 1.0/(2.0*T), N//2)
    X_fft=fft(X)
    X_fft[0]=0
    X_fft_magnitude=2.0/N*np.abs(X_fft[0:N//2])
    
    BinCounts=(N//2)//NBins
    
    SpecEnergy=[]
    #generating data columns label
    labels=[]
    for i in range(0,NBins):
        SpecEnergy.append(X_fft_magnitude[i*BinCounts:(i+1)*BinCounts].sum())
        labels.append('FFT'+str(i+1))
    
#    plt.figure(150)
#    plt.plot(np.arange(1,NBins+1),SpecEnergy)
#    plt.xlabel('Bin index')
#    plt.ylabel('Bin energy')
#    plt.figure(151)
#    plt.plot(FreqList, X_fft_magnitude)
#    plt.xlabel('Sample index')
#    plt.ylabel('Modal energy')
    
    return SpecEnergy,labels