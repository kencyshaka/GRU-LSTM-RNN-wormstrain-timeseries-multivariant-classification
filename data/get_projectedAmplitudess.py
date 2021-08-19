import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np
import h5py
import pickle
import math
import scipy.linalg as la
from sklearn.decomposition import PCA, FastICA
from mpi4py import MPI



def makeAngleArray(x,y):
    # initialize arrays
    numFrames,lengthX = x.shape
#    print(numFrames,lengthX)
    angleArray = np.zeros((numFrames, lengthX-1));
    meanAngles = np.zeros((numFrames, 1),dtype=np.complex_);

    #print("before****************")
    #print(meanAngles.shape, angleArray.shape)



    for i in range (numFrames):
        #calculate the x and y differences
        dX = np.diff(x[i])
        dY = np.diff(y[i])
#         print("the shape is ",angleArray[i].shape)
        angleArray[i] = np.arctan2(dY, dX);
        angleArray[i] = np.unwrap(angleArray[i])
        meanAngles[i] = np.mean(angleArray[i])

        angleArray[i] = angleArray[i] - meanAngles[i]

#     print("the values of the angle is ",angleArray[5])
#     print("the values of the mean angle is ",meanAngles[5])

    angleArray = angleArray[~np.isnan(angleArray).any(axis=1)]       #removes rows containing nan. values
    meanAngles = meanAngles[~np.isnan(meanAngles).any(axis=1)]       #removes rows containing nan. values
   # print("after ****************")
   # print(angleArray.shape, meanAngles.shape)

    return meanAngles,angleArray

def eigenWormProject(eigenWorms, angleArray, numEigWorms):
    an_array = np.empty((angleArray.shape[0], numEigWorms))

    an_array[:] = np.NaN

#     print(an_array)
    projectedAmps = an_array

    #Calculate time series of projections onto eigenworms
    for i in range(angleArray.shape[0]):
        rawAngles = angleArray[i,:]
        for j in range(numEigWorms):

            projectedAmps[i,j] = np.sum(eigenWorms[j,:]*rawAngles)
#    print("the projected Amps Shape",projectedAmps.shape)
    return projectedAmps

PCA = h5py.File("PCA_1N2_components.h5", "r")
ICA = h5py.File("ICA_1N2_components.h5", "r")
PCA = PCA['PCA']["component"] 
ICA = ICA['ICA']["component"] 
#labels = ['OW953','AQ2947','OW940','OW956','OW949','OW939','CB1112']
labels = ['OW939','CB1112']
for index in range(2):
  class_name = labels[index]
  print("processing the class", class_name)
  print("Trying to get the saved data skeleton")
  filenameX = "x_skeleton"+class_name+".h5"
  filenameY = "y_skeleton"+class_name+".h5"
  filenameX1 = "x_skeleton"+class_name+"1.h5"
  filenameY1 = "y_skeleton"+class_name+"1.h5"
  print("the files are ",filenameX)
  print("the class being eXplored is ",class_name)

  xf = h5py.File(filenameX, "r")
  yf = h5py.File(filenameY, "r")
  xf1 = h5py.File(filenameX1, "r")
  yf1 = h5py.File(filenameY1, "r")



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

  filenameX = "ICA_1"+class_name+"_features.h5"
  filenameY = "PCA_1"+class_name+"_features.h5"
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
# yf = h5py.File(filenameY, "r")
# print(xf)
  dsetr = xf["ICA"]

  if dsetr:
      print("datset accessible")
  else:
      print("dataset inaccessible")

  print("the length after retrieval",len(xf['/ICA']) )
  print(xf['ICA']['feature'].shape)
  
