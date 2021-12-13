import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np
import h5py
import pickle
import math
import scipy.linalg as la
from utils import *

PCA = h5py.File("features/PCA_N2_components.h5", "r")
ICA = h5py.File("features/ICA_N2_components.h5", "r")
PCA = PCA['PCA']["component"] 
ICA = ICA['ICA']["component"] 
labels = ['OW953','CB5','AQ2947','OW940','OW956','OW949','OW939','CB1112']
class_array = retrieve_pkl_file("class_int_to_word")

def checkclass(name):
    status = "False"
    for i in range(len(labels)):
        if name == labels[i]:
           status = "True"
    return status


for index in range(280,len(class_array)):
  print("at index *******",index)
  class_name = class_array[index]  
  if checkclass(class_name) == "False" :       
      
     print("processing the class", class_name)
     print("Trying to get the saved data skeleton")
     filenameX = "skeleton/x_skeleton"+class_name+".h5"
     filenameY = "skeleton/y_skeleton"+class_name+".h5"
 
     print("the files are ",filenameX)
     print("the class being eXplored is ",class_name)

     xf = h5py.File(filenameX, "r")
     yf = h5py.File(filenameY, "r")
  
     dsetr = xf["x"]
     print("the length is",len(dsetr))
     if dsetr:
         print("datset accessible")
     else:
         print("dataset inaccessible")

     print(xf['/x'][str(1)][0])
     print(yf['/y'][str(1)][0])



     print (" the length of the first loop",len(xf['/x']))
     for i in range(len(xf['/x'])): # length of the list , not we only need for 10 videos and concatenate together 

        print("in the loop",i)
        meanAngles, angleArray = makeAngleArray(xf['/x'][str(i)],yf['/y'][str(i)])

        if i == 0 :
                angleArrayCombined = angleArray
                meanAnglesCombined = meanAngles

        else:
               angleArrayCombined = np.concatenate((angleArrayCombined,angleArray), axis=0)
               meanAnglesCombined = np.concatenate((meanAnglesCombined,meanAngles), axis=0)
 
     ICAmatrix = eigenWormProject(ICA,angleArrayCombined,6)
     ICAfeatures = np.concatenate((ICAmatrix,meanAnglesCombined), axis=1)

     PCAmatrix = eigenWormProject(PCA,angleArrayCombined,4)
     PCAfeatures = np.concatenate((PCAmatrix,meanAnglesCombined), axis=1)

     print("the length of the PCA feature", len(PCAfeatures))
     #print("the shape of the feature", PCAfeaturesAQ2947[0].shape)

     print("saving the dataset  ***************************************************************")

     filenameX = "features1/ICA_"+class_name+"_features.h5"
     filenameY = "features1/PCA_"+class_name+"_features.h5"
     fx = h5py.File(filenameX, 'w')
     fy = h5py.File(filenameY, 'w')

     grpx=fx.create_group('ICA')
     grpx.create_dataset("feature",data=ICAfeatures)

     grpy=fy.create_group('PCA')
     grpy.create_dataset("feature",data=PCAfeatures)

     fx.close()
     fy.close()

     print("Trying to get the saved data")
     xf = h5py.File(filenameX, "r")
     

     dsetr = xf["ICA"]

     if dsetr:
        print("datset accessible")
     else:
        print("dataset inaccessible")

     print("the length after retrieval",len(xf['/ICA']) )
     print(xf['ICA']['feature'].shape)
  
