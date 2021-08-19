from imports import *
from utils import *
from config import *
from dataset import *
from model import *

import argparse
torch.cuda.empty_cache()
import gc
#del variables
gc.collect()

sweep_config = {
    'method': 'random', #grid, random
    'metric': {
      'name': 'train_f1score',
      'goal': 'maximize'   
    },
    'parameters': {
        'epochs': {
            'values': [5,10,15]
        },
        'batch_size': {
            'values': [32, 64,128]
        },
        'learn_rate': {
  	'distribution': 'uniform',
  	'min': 0.000001,
  	'max': 0.01
           # 'values': [ 0.0001, 0.0009]
        },
        'hidden_layer_size':{
            'values':[128,260,512]
        },
        'num_layers': {
            'values': [ 4, 6, 8]
        },
    },
    'early_terminate': {
        'type': 'hyperband',
        's': 2,
        'eta': 3,
        'max_iter': 15
    }
}

# Set up training arguments and parse
parser = argparse.ArgumentParser(description='Training the model using PCA/ICA features')
parser.add_argument(
    '--dimension_reduction',
    type=str,
    help='the optionos are PCA, ICA and CNN')
#parser.add_argument(
#    '--dimension_reduction', type=bool, default=True,
#    help='1 represents PCA and 0 represents ICA')

args = parser.parse_args()

dimension = args.dimension_reduction
print("the selected dimension reduction of the features is ",dimension)

if dimension == 'PCA':
    filename = FILENAME_PCA
    tag_feature = 'PCA'
    n_features = 5

else:
    filename = FILENAME_ICA
    tag_feature = 'ICA'
    n_features = 7


training_sequences, validation_sequences, test_sequences = data_splitter(filename,tag_feature)

sweep_id = wandb.sweep(sweep_config, project="last_GRU_"+str(SEQUENCE_LENGTH)+"_"+tag_feature+"_sweep")
print("the sweep_id is", sweep_id)


#def train():

defaults = dict(
      dropout=DROPOUT_RATIO,
      hidden_layer_size=HIDDEN_SIZE,
      num_layers=NUM_LAYERS,
      learn_rate=LR,
      batch_size = BATCH_SIZE,
      epochs=N_EPOCHS,
      )

wandb.init(config=defaults)
config = wandb.config
print("i am here")
# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %ii\\orboard --logdir ./PCA_lightning_logs

# Defining the ligthining model

class SequencePredictor2(pl.LightningModule):

         def __init__(self,n_features: int, n_classes: int, lr: float):
             super().__init__()
             self.model = SequenceModel(n_features,n_classes,config.hidden_layer_size,config.num_layers,config.dropout)
             self.criterion = nn.CrossEntropyLoss()
             self.lr = lr
             self.f1 = F1(num_classes=CLASS_SIZE)

         def forward(self,x, labels=None):

             output = self.model(x)
             loss = 0

             if labels is not None:
                 loss = self.criterion(output,labels)

             return loss, output

         def training_step(self, batch, batch_idx):   
        
#             print("I am called training step")
             sequences = batch["sequence"]
             labels = batch["label"]
             loss, outputs = self(sequences,labels)
             predictions = torch.argmax(outputs, dim=1)   # we want to calculate the accuracy of top 1 this can be changed
             step_accuracy = accuracy(predictions,labels)

             
             f1score = self.f1(predictions, labels)

             self.log("train_loss", loss, prog_bar=False,logger=True,sync_dist=True)
             self.log("train_accuracy", step_accuracy, prog_bar=True,logger=True,sync_dist=True)
             self.log("train_f1score",f1score,logger=True,sync_dist=True)

             return {"loss": loss, "accuracy": step_accuracy, "f1score": f1score }   # returns a dictionary

         def validation_step(self, batch, batch_idx):   #called everytime a validation needs to occur
       
             sequences = batch["sequence"]
             labels = batch["label"]
             loss, outputs = self(sequences,labels)
             predictions = torch.argmax(outputs, dim=1)
             step_accuracy = accuracy(predictions,labels)
             f1score  =  self.f1(predictions, labels)

             self.log("val_f1score",f1score,logger=True,sync_dist=True)
             self.log("val_loss", loss, prog_bar=True,logger=True,sync_dist=True)
             self.log("val_accuracy", step_accuracy, prog_bar=True,logger=True,sync_dist=True)

             return {"loss": loss, "accuracy": step_accuracy, "f1score": f1score}
      
         def test_step(self, batch, batch_idx):   #called everytime a test needs to occur
             sequences = batch["sequence"]
             labels = batch["label"]
             loss, outputs = self(sequences,labels)
             predictions = torch.argmax(outputs, dim=1)
             step_accuracy = accuracy(predictions,labels)

            
             f1score =  self.f1(predictions, labels)
             self.log("test_loss", loss, prog_bar=False,logger=True,sync_dist=True)
             self.log("test_accuracy", step_accuracy, prog_bar=False,logger=True,sync_dist=True)
             self.log("test_f1score",f1score ,prog_bar=True,logger=True,sync_dist=True)

             return {"loss": loss, "accuracy": step_accuracy, "f1score": f1score}

          
         def configure_optimizers(self):
             return optim.Adam(self.parameters(),lr=self.lr)

         def training_epoch_end(self, training_step_outputs):
             avg_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
             avg_f1score = torch.stack([x['f1score'] for x in training_step_outputs]).mean()
             avg_accuracy = torch.stack([x['accuracy'] for x in training_step_outputs]).mean()

             self.log("avg_train_loss", avg_loss, logger=True)
             self.log("avg_train_f1score",avg_f1score, logger=True)
             self.log("avg_train_accuracy",avg_accuracy, logger=True)



         def validation_epoch_end(self, validation_step_outputs):
             avg_loss = torch.stack([x['loss'] for x in validation_step_outputs]).mean()
             avg_f1score = torch.stack([x['f1score'] for x in validation_step_outputs]).mean()
             avg_accuracy = torch.stack([x['accuracy'] for x in validation_step_outputs]).mean()

             self.log("avg_val_loss", avg_loss, logger=True)
             self.log("avg_val_f1score",avg_f1score, logger=True)
             self.log("avg_val_accuracy",avg_accuracy, logger=True)
 
model = SequencePredictor2(n_features,CLASS_SIZE,config.learn_rate)
data_module = SkeletonDatasetModule(training_sequences,validation_sequences,test_sequences,config.batch_size)
print("about to define the models")

def train(model, datamodule): 
  torch.multiprocessing.freeze_support() 
  checkpoint_callback = ModelCheckpoint(
      dirpath = tag_feature+"checkpoints",
      filename = tag_feature+"best_checkpoint",
      save_top_k = 1,
      verbose = True,
      monitor = "train_f1score",
      mode = "max"
     )  

  early_stop_callback = EarlyStopping(
  #  filename ="PCA_earlystopping_checkpoint"
     monitor="train_f1score",
     min_delta=0.00,
     patience=3,
     verbose=True,
     mode="max"
     )
#wandb login

  wandb_logger = WandbLogger(save_dir=str(SEQUENCE_LENGTH)+"_"+tag_feature+"lightning_logs")

#logger = TensorBoardLogger(tag_feature+"lightning_logs", name=tag_feature+"experiment_mini")

  trainer = pl.Trainer(
      logger=wandb_logger,
      callbacks=[early_stop_callback],
      checkpoint_callback=checkpoint_callback,
      max_epochs=config.epochs,
      gpus = 1,
      progress_bar_refresh_rate=30

     )  

  trainer.fit(model, data_module)
  
  
  
if __name__ == "__main__":
  # torch.multiprocessing.freeze_support() 
   wandb.agent(sweep_id, train(model,data_module))
