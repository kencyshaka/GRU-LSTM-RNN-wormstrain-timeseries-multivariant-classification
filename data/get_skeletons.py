import h5py
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np
import pickle
import os
import collections



list_path = []

basepath = 'dataset1/'

for entry in os.listdir(basepath):
    path = os.path.join(basepath, entry)
    if os.path.isdir(path):
        for filename in os.listdir(path):
            list_path.append(path+"/"+filename)

def class_to_int(class_list):
    
    class_list = list(set(class_list))   # removing the duplicates
    print("the number of classes present in the dataset",len(class_list))
    
    class_int = list(range(0, len(class_list)))
    # print(class_list)
    print(class_int)
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

\


features= []
class_list = []
labels = ['N2','OW953','AQ2947','OW940','OW956','OW949','OW939','CB1112']

print("the length is", len(list_path))

for i in range(len(list_path[10235:])):
    s = h5py.File(list_path[i],mode='r')
    print(i,"********")
    if get_strain_name(s) == 'CB1112':
    	class_list.append(get_strain_name(s))
    	features.append(s)
    
print("the length of the features", len(features))



counter = collections.Counter(class_list)

print(counter)


df = pd.DataFrame(features)

print(df.head())


skeleton_dataframe_collectionx = [] 
skeleton_dataframe_collectiony = [] 

for i in range(len(features)):  
    print(i)
    n = df['coordinates'][i]['skeletons'][()]
    datax,datay = get_skeleton_array(n)
    
    skeleton_dataframe_collectionx.append(datax)
    skeleton_dataframe_collectiony.append(datay)
    print(i)
    
print("before saving")   

filenameX = "x_skeletonCB11121.h5"    #filename of the x features of a particular worm strain
filenameY = "y_skeletonCB11121.h5"    #filename of the x features of a particular worm strain
fx = h5py.File(filenameX, 'w')
fy = h5py.File(filenameY, 'w')

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


print(xf['/x'][str(1)][0])
print(yf['/y'][str(1)][0])
