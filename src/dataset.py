
from imports import *
from utils import *
from config import *


# Dataset Class

class SkeletonDataset(Dataset):
    def __init__ (self,sequence):
        self.sequence = sequence
    
    def __len__(self):
        return len(self.sequence)
    
    def __getitem__(self,idx):
      
        sequence, label = self.sequence[idx]
        #print("the sequence", sequence)
        return dict(
        sequence =torch.Tensor(sequence),
        label = torch.tensor(label).long()    
        )

class SkeletonDatasetModule(pl.LightningDataModule):
    
    def __init__(self,train_sequence,val_sequence,test_sequence,batch_size):
        super().__init__()
        self.train_sequence = train_sequence
        self.val_sequence = val_sequence
        self.test_sequence = test_sequence
        self.batch_size = batch_size
     
    def setup(self,stage=None):
        self.train_dataset = SkeletonDataset(self.train_sequence)
        self.test_dataset = SkeletonDataset(self.test_sequence)
        self.val_dataset = SkeletonDataset(self.val_sequence)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers=NUM_WORKERS
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers=NUM_WORKERS
        ) 
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers=NUM_WORKERS
        )

        


