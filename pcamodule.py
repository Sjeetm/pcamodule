# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:51:13 2018

@author: Subhajeet
"""

def data_clean(df):
    import numpy as np
    df=df.select_dtypes(include=np.number) #selecting numeric columns
    return df

def data_norm(df):
    import numpy as np
    #normalising the dataset
    d_m=np.mean(df)
    d_s=np.std(df)
    d_n=(df-d_m)/d_s
    d_n=d_n.dropna(axis=1)
    return d_n

def data_eig(df):
    import numpy as np
    eig=np.linalg.eig
    d_c=np.cov(df,rowvar=False) #computing covariance matrix for normalised dataset
    eva,evec=eig(d_c) #calculating eigen values and its corresponding eigenvectors
    #sorting eigenvectors in decreasing value of eigenvalue
    j=eva.argsort()[::-1]
    ev=evec[:,j]
    return ev

def error_plot(df,v):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error as mse #for computing mean absolute error
    err=[]
    for i in range(len(v)):   
        gt=np.asmatrix(v[:,0:i+1]).T #taking Principal Components
        t=np.matmul(gt.T,np.matmul(gt,np.asmatrix(df).T))
        err.append(mse(np.asmatrix(df).T,t)) #computing sum squared error
    #plotting the errors
    n=min(df.shape)+1
    plt.title('Error Graph')
    plt.xlabel('PC')
    plt.ylabel('SSE')
    return plt.plot(list(range(1,n)),err,color='blue', marker='o',markerfacecolor='red', markersize=8)

def pca(df):
    data=data_clean(df)
    norm_data=data_norm(data)
    v=data_eig(norm_data)
    error_plot(norm_data,v)