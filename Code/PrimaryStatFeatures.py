def PrimaryFeatureExtractor(X):
    from scipy.stats import kurtosis,skew
    import numpy as np
    Y=[]
    rms=np.sqrt(np.mean(np.square(X)))
    #Crest Factor: Crest factor is a parameter of a waveform,
    #such as alternating current or sound,
    #showing the ratio of peak values to the effective value.
    CrestFactor=abs(X).max()/rms
    #shape factor is defined as the ratio of the signal's RMS value to its absolute mean
    ShapeFactor=rms/(abs(X).mean())
    Y.append([abs(X).mean(),X.min(),X.max(),X.std(),rms,skew(X),kurtosis(X),CrestFactor,ShapeFactor])

    return Y