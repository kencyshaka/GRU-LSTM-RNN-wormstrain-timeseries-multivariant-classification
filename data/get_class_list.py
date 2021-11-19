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



list_path = []
features= []
class_list = []

for entry in os.listdir(BASEPATH):
    path = os.path.join(BASEPATH, entry)
    if os.path.isdir(path):
        for filename in os.listdir(path):
            list_path.append(path+"/"+filename)
            file = h5py.File(path+"/"+filename,mode='r')
            class_list.append(get_strain_name(file))
            file.close()

counter = collections.Counter(class_list)  #counting number of videos per class strain

print(counter)
class_word_to_int, class_int_to_word = class_to_int(class_list)
print(class_int_to_word)

saving_file_pkl(PATHLIST,list_path)
saving_file_pkl(CLASS_WORD_TO_INT,class_word_to_int)
saving_file_pkl(CLASS_INT_TO_WORD,class_int_to_word)


