
from imports import *

#path to the files of worm strain reduced features
FILENAME_PCA = ["/nobackup/sc20ms/experiment/skeleton/trial/data/PCA_1AQ2947_features.h5",
                "/nobackup/sc20ms/experiment/skeleton/trial/data/PCA_1N2_features.h5",
"/nobackup/sc20ms/experiment/skeleton/trial/data/PCA_1CB1112_features.h5",
"/nobackup/sc20ms/experiment/skeleton/trial/data/PCA_1OW939_features.h5",
"/nobackup/sc20ms/experiment/skeleton/trial/data/PCA_1OW949_features.h5",
"/nobackup/sc20ms/experiment/skeleton/trial/data/PCA_1OW956_features.h5",
"/nobackup/sc20ms/experiment/skeleton/trial/data/PCA_1OW940_features.h5",
"/nobackup/sc20ms/experiment/skeleton/trial/data/PCA_1OW953_features.h5"]

FILENAME_ICA = ["/nobackup/sc20ms/experiment/skeleton/trial/data/ICA_1AQ2947_features.h5",
   "/nobackup/sc20ms/experiment/skeleton/trial/data/ICA_1N2_features.h5",
"/nobackup/sc20ms/experiment/skeleton/trial/data/ICA_1CB1112_features.h5",
"/nobackup/sc20ms/experiment/skeleton/trial/data/ICA_1OW939_features.h5",
"/nobackup/sc20ms/experiment/skeleton/trial/data/ICA_1OW949_features.h5",
"/nobackup/sc20ms/experiment/skeleton/trial/data/ICA_1OW956_features.h5",
"/nobackup/sc20ms/experiment/skeleton/trial/data/ICA_1OW940_features.h5",
"/nobackup/sc20ms/experiment/skeleton/trial/data/ICA_1OW953_features.h5"]

FILENAME = "/nobackup/sc20ms/experiment/skeleton/trial/data/features/"

NORM_MEAN = [0.52283615, 0.47988218, 0.40605107]
NORM_STD = [0.29770654, 0.2888402, 0.31178293]

#i Define the class labels
CLASS_INDEX = [0,1,2,3,4,5,6,7,8]
CLASS_NAMES = ["AQ2947","N2","CB1112","CB5","OW939","OW949","OW956","OW940","OW953"]
CLASS_SIZE = len(CLASS_NAMES)


# variables for pre-processing the dataset
SEQUENCE_LENGTH =600
SPLIT_RATIO = 0.2

#variables for defining datamodules
NUM_WORKERS = 12  # or cpu_count()
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 64



#variables for trainign model
N_EPOCHS = 5
HIDDEN_SIZE = 512
NUM_LAYERS = 5
LR = 0.0001
DROPOUT_RATIO = 0.25

print("the class size is",CLASS_SIZE)
print("availablee GPU :",AVAIL_GPUS)
print("the batchsize is :",BATCH_SIZE)
print("number of epoch",N_EPOCHS)
