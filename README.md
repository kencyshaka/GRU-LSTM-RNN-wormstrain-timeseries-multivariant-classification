# GRU-LSTM-RNN-worm strain-time series-multivariant-classification
This project is an effort in the behavioural quantification field by examining automatic feature extraction and identifying C.elegans strain using deep learning techniques. The study demonstrates how LSTM and GRU can stand on their own as worm strain classifiers and the effect of dimension reduction on the model performance.

The implementation is in python, PyTorch, pytorch-lightning and wandb.

### To download the dataset
 > download_zenosh.sh  download_skeleton.txt <specify output file> 
 
### To run the experiment 
 >python src/training.py  <specify the dimension reduction method (PCA|ICA)>
