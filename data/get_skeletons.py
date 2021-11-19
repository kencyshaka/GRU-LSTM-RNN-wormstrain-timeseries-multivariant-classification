import h5py
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np
import pickle
import os
import collections

from utils import *
from config import *

def run():

    list_path = output = retrieve_pkl_file(PATHLIST)
    class_list = retrieve_pkl_file(CLASS_INT_TO_WORD)
    features = []
    

    for index in range(len(class_list)):
        print("the class being extracted is:",class_list[index])
        print("the number of all videos is", len(list_path))
        
        skeleton_dataframe_collectionx = []
        skeleton_dataframe_collectiony = []

        for i in range(len(list_path)):
            s = h5py.File(list_path[i],mode='r')
           
            print(i,"********")
            if get_strain_name(s) == class_list[index]:
                x, y = get_skeleton_array(s['coordinates/skeletons'])
                skeleton_dataframe_collectionx.append(x)
                skeleton_dataframe_collectiony.append(y)
               
            s.close()
        
        print("the length of the x features", len(skeleton_dataframe_collectionx))
        #print("the length of the x shape", skeleton_dataframe_collectionx[0])
        
        filenameX = "skeleton/x_skeleton"+class_list[index]+".h5"    #filename of the x features of a particular worm strain
        filenameY = "skeleton/y_skeleton"+class_list[index]+".h5"    #filename of the x features of a particular worm strain
        fx = h5py.File(filenameX, 'w')
        fy = h5py.File(filenameY, 'w')

        print("************ saving ",class_list[index])
        
        grpx=fx.create_group('x')
        for i,list in enumerate(skeleton_dataframe_collectionx):
            grpx.create_dataset(str(i),data=list)

        grpy=fy.create_group('y')
        for i,list in enumerate(skeleton_dataframe_collectiony):
            grpy.create_dataset(str(i),data=list)
            

        fx.close()
        fy.close()


        print("Trying to get the saved data")

        xf = h5py.File(filenameX, "r")
        yf = h5py.File(filenameY, "r")
        print(xf)
        dsetr = xf["x"]

        if dsetr:
            print("datset accessible")
        else:
            print("dataset inaccessible")

        print(xf['/x'][str(0)][0])
        print(yf['/y'][str(0)][0])
        
        xf.close()
        yf.close()

if __name__ == '__main__':
    run()
