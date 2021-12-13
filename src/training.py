
from imports import *
from utils import *
from config import *
from dataset import *
from model import *
import sys
import argparse

torch.cuda.empty_cache()
import gc
#del variables
gc.collect()
torch.cuda.memory_summary(device=None, abbreviated=False)


# Set up training arguments and parse
parser = argparse.ArgumentParser(description='Training the model using PCA/ICA features')
parser.add_argument(
    '--dimension_reduction',
    type=str,
    help='the optionos are PCA and ICA ')

    
args = parser.parse_args()

dimension = args.dimension_reduction
print("the selected dimension reduction of the features is ",dimension)

if dimension == 'PCA':
    filename = FILENAME
    tag_feature = 'PCA'
    n_features = 5

else:
    filename = FILENAME
    tag_feature = 'ICA'
    n_features = 7
    

if __name__ == "__main__":
   training_sequences, validation_sequences, test_sequences = data_splitter(filename,tag_feature)


   model = SequencePredictor(n_features,CLASS_SIZE,LR)
   data_module = SkeletonDatasetModule(training_sequences,validation_sequences,test_sequences,BATCH_SIZE)

   checkpoint_callback = ModelCheckpoint(
      dirpath = tag_feature+"checkpoints",
      filename = tag_feature+"best_checkpoint",
      save_top_k = 1,
      verbose = True,
      monitor = "val_f1score",
      mode = "max"
   )

   early_stop_callback = EarlyStopping(
     monitor="val_f1score",
     min_delta=0.00,
     patience=2,
     verbose=True,
     mode="max"
   )


#wandb login

   wandb_logger = WandbLogger(save_dir=str(SEQUENCE_LENGTH)+"_"+tag_feature+"lightning_logs", project="comparison_experiment_baseline")

#logger = TensorBoardLogger(tag_feature+"lightning_logs", name=tag_feature+"experiment_mini")

   trainer = pl.Trainer(
      logger=wandb_logger,
      callbacks=[checkpoint_callback],
      checkpoint_callback=True,
      #checkpoint_callback=checkpoint_callback,    
      max_epochs=N_EPOCHS,
      gpus = 1,
      progress_bar_refresh_rate=30
    
    )  

   print("***********************Training the model using "+str(SEQUENCE_LENGTH)+"_"+tag_feature+" features ************************")
   trainer.fit(model, data_module)


   print("***********************Testing the model using "+str(SEQUENCE_LENGTH)+"_"+tag_feature+" features ************************")

   trainer.test() #ckpt_path=None
   trainer.save_checkpoint("/nobackup/sc20ms/experiment/skeleton/saved_models/data_GRU_1_"+str(SEQUENCE_LENGTH)+"_"+tag_feature+"_early_stoping.ckpt")

  

   trained_model = SequencePredictor.load_from_checkpoint(
     #trainer.checkpoint_callback.best_model_path,
     checkpoint_path="/nobackup/sc20ms/experiment/skeleton/saved_models/data_GRU_1_"+str(SEQUENCE_LENGTH)+"_"+tag_feature+"_early_stoping.ckpt",
     n_features = n_features,
     n_classes = CLASS_SIZE,
     lr = LR
   )  
   trained_model.eval()
   trained_model.freeze()   #only needed for inference

   # create the test dataset
   # load the test set

   test_dataset = SkeletonDataset(test_sequences)

   predictions = []
   labels = []
   print("the length of the test set is",len(test_dataset))
   step = 0
   for item in tqdm(test_dataset):
       print("in step",step)
       sequence = item["sequence"]
       label = item["label"]
    
       _, output = trained_model(sequence.unsqueeze(dim=0))
       prediction = torch.argmax(output, dim=1)
       step = step + 1
    #if step == 8:
    #   break	
       predictions.append(prediction[0].item())
       labels.append(label.item())
       #print("theprediction is",predictions)
       #print("the labels are",labels)
    
   wandb.log({"final"+str(SEQUENCE_LENGTH)+"_"+tag_feature+"conf_mat" : wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=labels,
                        preds=predictions,
                        class_names=CLASS_NAMES)})


   #evaluating the result
   plot_confusion_matrix(predictions,labels,tag_feature)

   report = classification_report(labels, predictions)
   print("the classification report is")
   print(report)

   sys.exit()
