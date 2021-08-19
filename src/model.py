
from imports import *
from config import *
from dataset import *

        
#Building the sequence Model

class SequenceModel(nn.Module):
    
    def __init__(self,n_features, n_classes,n_hidden, n_layers,dropout):
        super().__init__()
        
        self.lstm = nn.LSTM(                        #can be changed to GRU or RNN
            input_size = n_features,
            hidden_size = n_hidden,
            num_layers = n_layers,
            batch_first = True,
            dropout = dropout
        )
        
        self.classifier = nn.Linear(n_hidden,n_classes)
        
    def forward(self,x):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)
#        out, _ = self.lstm(x)                 #when using RNN or GRU
#        out = out[:, -1, :]                   #when using RNN or GRU
        out = hidden[-1]
         
        return self.classifier(out)


# Defining the ligthining model

class SequencePredictor(pl.LightningModule):
    
    def __init__(self,n_features: int, n_classes: int, lr: float):
        super().__init__()
        self.model = SequenceModel(n_features,n_classes,HIDDEN_SIZE,NUM_LAYERS,DROPOUT_RATIO)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.f1 = F1(num_classes=CLASS_SIZE)     
    def forward(self,x, labels=None):
        
        output = self.model(x)
        loss = 0
        
        if labels is not None:
            loss = self.criterion(output,labels)
        
        return loss, output
    
    def training_step(self, batch, batch_idx):   #called everytime a training needs to occur
        # print("I am called ttraining step")
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, outputs = self(sequences,labels)
        predictions = torch.argmax(outputs, dim=1)   # we want to calculate the accuracy of top 1 this can be changed
        step_accuracy = accuracy(predictions,labels)
        
        f1score = self.f1(predictions, labels)    
        
        self.log("train_loss", loss, prog_bar=False,logger=True)
        self.log("train_accuracy", step_accuracy, prog_bar=True,logger=True)
        self.log("train_f1score",f1score,logger=True)
        
        return {"loss": loss, "accuracy": step_accuracy, "f1score": f1score }   # returns a dictionary
    
    def validation_step(self, batch, batch_idx):   #called everytime a validation needs to occur
        # print("I am called validation step")
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, outputs = self(sequences,labels)
        predictions = torch.argmax(outputs, dim=1)   
        step_accuracy = accuracy(predictions,labels)
        
        f1score = self.f1(predictions, labels)

        self.log("val_f1score",f1score,logger=True)
        self.log("val_loss", loss, prog_bar=True,logger=True)
        self.log("val_accuracy", step_accuracy, prog_bar=True,logger=True)
        
        return {"loss": loss, "accuracy": step_accuracy, "f1score": f1score}  
    
    def test_step(self, batch, batch_idx):   #called everytime a test needs to occur
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, outputs = self(sequences,labels)
        predictions = torch.argmax(outputs, dim=1)   
        step_accuracy = accuracy(predictions,labels)
        
        f1score = self.f1(predictions, labels)        
        self.log("test_loss", loss, prog_bar=False,logger=True)
        self.log("test_accuracy", step_accuracy, prog_bar=False,logger=True)
        self.log("test_f1score",f1score ,prog_bar=True,logger=True)

        return {"loss": loss, "accuracy": step_accuracy, "f1score": f1score } 
    
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

    def test_epoch_end(self, test_step_outputs):
          avg_loss = torch.stack([x['loss'] for x in test_step_outputs]).mean()
          avg_f1score = torch.stack([x['f1score'] for x in test_step_outputs]).mean()
          avg_accuracy = torch.stack([x['accuracy'] for x in test_step_outputs]).mean()

          self.log("avg_test_loss", avg_loss, logger=True)
          self.log("avg_test_f1score",avg_f1score, logger=True)
          self.log("avg_test_accuracy",avg_accuracy, logger=True) 
