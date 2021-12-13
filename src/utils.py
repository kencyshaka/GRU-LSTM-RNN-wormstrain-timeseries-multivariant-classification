
from imports import *
from config import *
from collections import Counter

np.set_printoptions(precision=2)

# A function to retrieve files stored in pkl
def retrieve_pkl_file(filename):
    a_file = open(filename+".pkl", "rb")
    output = pickle.load(a_file)

    return output

#getting the class names
def get_class_names(filename):
    class_to_word = retrieve_pkl_file(filename)
    return class_to_word.keys()
 
#getting the class index
def get_class_index(filename):
    class_to_index = retrieve_pkl_file(filename)
    return class_to_index.keys()

def data_splitter(filename,tag_feature):
    sequences = []
    print("length of the files",len(filename))
    
    for i in range(CLASS_SIZE):
        
        features = h5py.File(FILENAME+tag_features+"_"+CLASS_NAMES[i]+"_features.h5", "r")
        label = CLASS_INDEX[i]
        
        print("inside with class ", label)
        if features[tag_feature]['feature']:
           print("datset accessible")
        else:
           print("dataset inaccessible")
        
        data = features[tag_feature]['feature'] 
        
           
        
        
        #print("the data is", data[()])
        print(" the inside data is", data.shape)
        print(" the inside data is", len(data))
        
        X=data[:] 
        data = X.astype(np.float32)
        print(data[0:2])     
        ylabel = label
        print ("the shape of the data",data.shape)
        print("also",len(data))
        segment = int((len(data))/SEQUENCE_LENGTH)
        
        stop = 0
        sampled = data[0:30:3]
        print("shape of sub sample one",sampled)
#        new = data[str(i)][()]
        print("the number of segments  are",segment)   
        for j in range(segment):
                
                start =  stop
                stop = start  + SEQUENCE_LENGTH
                sub_sampled = data[start:stop:3]
                print("shape of sub sample one",start,stop,sub_sampled.shape)
        #        sub = int(SEQUENCE_LENGTH/30)
        #        for s in range(sub):
        #            start = start +30
        #            stop = stop + 30
        #            sample = data[start : stop: 3]
        #            print("looing sample",sample.shape,start,stop)
        #            sub_sampled = np.concatenate((sub_sampled, sample), axis=0)
        #        print("start is ", start,stop)
        #        print("the subsampled shape ", sub_sampled.shape)
                sequences.append((sub_sampled,ylabel))
                
        print("done with class ",CLASS_INDEX[i])
    sequences = np.array(sequences) 
    print(" split into training and test set")
    training, test_sequences = train_test_split(sequences, test_size=SPLIT_RATIO, random_state=42, shuffle=True)
    training_sequences, validation_sequences = train_test_split(training, test_size=SPLIT_RATIO, random_state=42, shuffle=True)

    print("the length of training, validatiion and test sequence",training_sequences.shape,validation_sequences.shape,test_sequences.shape)
    
    c_list = Counter(training_sequences[:,1])
    print("training",c_list) 
   
    ic_list = Counter(validation_sequences[:,1])
    print("validation",ic_list)
    
    dc_list = Counter(test_sequences[:,1])
    print("training",dc_list)	 
    return training_sequences, validation_sequences, test_sequences
    
    
def classification_report_csv(report):
    report_data = []
    lines = report.split('\n')
    print(lines[2:-3])
    for line in lines[2:-4]:
        print(line)
        row = {}
        row_data = line.split('      ')
        print(len(row_data))
        print(row_data)
        print(row_data[1])
        

        row['class'] = row_data[1]
        row['precision'] = float(row_data[2])
        row['recall'] = float(row_data[3])
        row['f1_score'] = float(row_data[4])
        row['support'] = float(row_data[5])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv('classification_report.csv', index = False)
    

def plot_confusion_matrix(y_pred, y_true, tag_feature,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    title=str(SEQUENCE_LENGTH)+" "+tag_feature+" Confusion matrix"
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(CLASS_SIZE)
    plt.xticks(tick_marks, CLASS_INDEX, rotation=45)
    plt.yticks(tick_marks, CLASS_INDEX)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("/nobackup/sc20ms/Trial/figure/"+str(SEQUENCE_LENGTH)+"_"+tag_feature+"cm.pdf")

    
