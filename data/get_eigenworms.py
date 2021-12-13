import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np
import h5py
import pickle
import math
import scipy.linalg as la
from sklearn.decomposition import PCA, FastICA
from utils import *


def run():

    xf = {}
    yf = {}
    for i in range (5):
 
        xf[i] = h5py.File("skeleton/x_skeletonN2"+str(i)+".h5", "r")
        yf[i]= h5py.File("skeleton/y_skeletonN2"+str(i)+".h5", "r")

        dsetr = xf[i]["x"]
        print(dsetr)

    for file in range (0,1):
        for i in range(len(xf[file]['/x'])):   # all  videos concatenated together 

            print("in the loop",i)
            meanAngles, angleArray = makeAngleArray(xf[file]['/x'][str(i)],yf[file]['/y'][str(i)])

            if i == 0 :
                angleArrayCombined = angleArray
                meanAnglesCombined = meanAngles
         
            else:
               angleArrayCombined = np.concatenate((angleArrayCombined,angleArray), axis=0)
               meanAnglesCombined = np.concatenate((meanAnglesCombined,meanAngles), axis=0)

    print("the initial combined angle",angleArrayCombined.shape)
    
    for file in range (1,5):
    
        for i in range(len(xf[file]['/x'])):   

            print("in the loop",i)
            meanAngles, angleArray = makeAngleArray(xf[file]['/x'][str(i)],yf[file]['/y'][str(i)])
        
            angleArrayCombined = np.concatenate((angleArrayCombined,angleArray), axis=0)
            meanAnglesCombined = np.concatenate((meanAnglesCombined,meanAngles), axis=0) 
    
    ICAvectors,ICAmatrix = ICA_eigenWorms(angleArrayCombined,6)
    ICAfeatures = np.concatenate((ICAmatrix,meanAnglesCombined), axis=1)

    values, PCAvectors,PCAmatrix,variance = PCA_eigenWorms(angleArrayCombined,4)
    PCAfeatures = np.concatenate((PCAmatrix,meanAnglesCombined), axis=1)
    
    print("the shape of the N2 PCA features", PCAfeatures.shape)
    print("the shape of the N2 PCA vectors", PCAvectors.shape)
    print("the shape of the N2 ICA features", ICAfeatures.shape)
    print("the shape of the N2 ICA vector", ICAvectors.shape)
    print("the eigen worms shape PCA",values.shape)

    print("saving the dataset  ***************************************************************")
    class_name =  "N2"
    
    saving_file_pkl("PCA_N2_eigenworms",values)
    saving_file_pkl("PCA_N2_variance",variance)
    
    filenameX = "features/ICA_"+class_name+"_features.h5"
    filenameY = "features/PCA_"+class_name+"_features.h5"
    fx = h5py.File(filenameX, 'w')
    fy = h5py.File(filenameY, 'w')

    grpx=fx.create_group('ICA')
    grpx.create_dataset("feature",data=ICAfeatures)

    grpy=fy.create_group('PCA')
    grpy.create_dataset("feature",data=PCAfeatures)

    fx.close()
    fy.close()

    filename1 = "features/ICA_"+class_name+"_components.h5"
    filename2 = "features/PCA_"+class_name+"_components.h5"
    ica = h5py.File(filename1, 'w')
    pca = h5py.File(filename2, 'w')

    grpx=ica.create_group('ICA')
    grpx.create_dataset("component",data=ICAvectors)

    grpy=pca.create_group('PCA')
    grpy.create_dataset("component",data=PCAvectors)

    ica.close()
    pca.close()

    print("Trying to get the saved data")
    xf = h5py.File(filenameX, "r")
    
    dsetr = xf["ICA"]

    if dsetr:
        print("datset accessible")
    else:
        print("dataset inaccessible")

    print("the length after retrieval",len(xf['/ICA']) )
    print(xf['/ICA']['feature'][()].shape)
    print("the length after retrieval",len(xf['/ICA']['feature']) )


if __name__ == '__main__':
    run()
