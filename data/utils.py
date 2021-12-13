import h5py
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np
import pickle
import os
import collections
from sklearn.decomposition import PCA, FastICA

# A function to convert the class into intergers
def class_to_int(class_list):
    
    #class_list = list(set(class_list))   # removing the duplicates
    class_list = set(class_list)  # removing the duplicates
    print("the number of classes present in the dataset",len(class_list))
    
    class_int = range(0, len(class_list))
    # print(class_list)
    #print(class_int)
    zipbObj1 = zip(class_list, class_int)
    zipbObj2 = zip(class_int, class_list)
    return dict(zipbObj1), dict(zipbObj2)


# A function to extract the strain name
def get_strain_name(filename):
    s = filename['experiment_info']
    list = s[()].split()
    name = b'"strain":'
    strain_name = ""
    for i in range(len(list)):
        if list[i] == name:
            strain_name = list[i+1]  
            strain_name = format_to_string(strain_name) 
            strain_name = clean(strain_name)
            break
    return strain_name

# A function to convert and clean the string
def format_to_string(text):
    text=str(text, 'utf-8')
    return text

def clean(text):
    text = re.sub('[",<!@#$)(]', '', text)
    return text

# A function to save a file in pkl format
def saving_file_pkl(filename,data):
    f = open(filename+".pkl","wb")
    pickle.dump(data,f)
    f.close
    
    
# A function to retrieve files stored in pkl
def retrieve_pkl_file(filename):
    a_file = open(filename+".pkl", "rb")
    output = pickle.load(a_file)
    
    return output

# to check the datatype of the hdf5 file
def visitor_func(name, node):
    if isinstance(node, h5py.Group):
        print(node.name, 'is a Group')
    elif isinstance(node, h5py.Dataset):
        if (node.dtype == 'object') :
            print (node.name, 'is an object Dataset')
            
        else:
            print(node.name, 'is a Dataset', node.dtype)
            
    else:
        print(node.name, 'is an unknown type') 
        
def get_skeleton_array(f):
    
#     f = feature['coordinates/skeletons']
    length = len(f)
    print("the length is",length)
    tx = np.zeros((length,49))
    ty = np.zeros((length,49))
    
    print("the shape is ",tx.shape)
    for i in range(length):
        for j in range(49):
            tx[i,j] = f[i][j][0]
            ty[i,j] = f[i][j][1]
    
    return tx, ty        

def PCA_eigenWorms(angleArrayCombined,numEigWorms):

    angleArrayCombined = angleArrayCombined[~np.isnan(angleArrayCombined).any(axis=1)]       #removes rows containing nan. values

    pca = PCA(n_components=numEigWorms)
    pca.fit(angleArrayCombined)
    eigenWorms = pca.components_
    eigenValues = pca.explained_variance_
    matrix = pca.fit_transform(angleArrayCombined)

    variance = sum(pca.explained_variance_ratio_)
    print("the variance is",variance)
    return eigenValues,eigenWorms,matrix,variance

def ICA_eigenWorms(angleArrayCombined,numEigWorms):

    angleArrayCombined = angleArrayCombined[~np.isnan(angleArrayCombined).any(axis=1)]       #removes rows containing nan. values
    print(angleArrayCombined.shape)
    ica = FastICA(n_components=numEigWorms)
    ica.fit(angleArrayCombined)
    eigenWorms = ica.components_
    matrix = ica.fit_transform(angleArrayCombined)
    print("ICA variation",ica.get_params)
    return eigenWorms,matrix

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
