# CS6910 Assignment 3
The provided code snippet demonstrates how to implement a sequence-to-sequence (Seq2Seq) model using PyTorch. It begins by preparing and preprocessing a dataset consisting of English-Hindi word pairs. The code defines both an Encoder and a Decoder class, utilizing RNN cells such as LSTM, GRU, or vanilla RNN. The Seq2Seq model is then initialized with these components. Finally, the forward pass is executed, where the source sequence is encoded, and the target sequence is generated through decoding.

# 1)DL_Assignment3.ipynb and train.py

## DL_Assignment3.ipynb
- This file contains The code for all the questions given in assignment.
- This is ipynb file.
- Thus run each cell one by one to see the output.
- It contains code for both seq2seq model (with attention and without attention)

## train.py
-   This is a python file
-   You can set parameters value of your choice .
-   You can give arguments as well.
-   **Before running ensure to change datapaths in variable:(give path to train.csv,val.csv,test.csv)**
    - trainFilepath --> path to hin_train.csv
    - valFilePath   --> path to hin_valid.csv
    - testFilePath  --> path to hin_test.csv
-   I have run the file on kaggle thus data paths given for dataset are in that way.(You need to change it accordingly)
-   train() and test() methods are called at the end of the file thus the model will be trained and tested and the accuracies will be printed as well as a sweep run will be created.

### If you want to Run train.py on kaggle*
-  First upload dataset on kaggle.
-  then upload train.py file as well  
-  run code -> !pip insatll argparse 
-   Before running ensure to change datapaths in variable:(give path to train.csv,val.csv,test.csv)
    - trainFilepath
    - valFilePath
    - testFilePath
-  To run now -> !python $path_to_uploaded_train.py file(on kaggle)

### If you want to Run train.py on CMD*
-  When needed Will have to install some packages.
-   Before running ensure to change datapaths in variable:(give path to train.csv,val.csv,test.csv)
    - trainFilepath
    - valFilePath
    - testFilePath 

## Parameter choices which can be pass as command line arguments
-   Cell Type : '-ct','--cell_type', choices=["RNN", "GRU", "LSTM"]
-   Attention : '-at','--attention', choices=[True,False]
-   Epoch : '-e', '--epochs',  choices=[1 , 2 ,10] etc
-   Drop Out : '-do','--drop_out', choices=[0.3, 0.5 , 0] etc
-   Learning rate : '-lr', '--learning_rate', choices=[0.001 ,0.005] etc
-   Embedding size : '-es','--embedding_size', choices=[32,64,128,256] etc
-   Hidden Layer Size : '-hs','--hidden_layer_size', choices=[32,64,128,256,512] etc
-   Encoder Layers : '-ne','--encoder_layers', choices=[1 , 2 ,3] etc
-   Decoder Layers : '-nd','--decoder_layers',choices=[1 , 2 ,3] etc
-   Bidirectional : '-bd','--bidirectional',choices=[True,False]
**Default set to my best configuration for attention model**


## Some sections in the code

### Data loading and preprocessing: 
- contain various functions which will basically load the data from csv file.
- Map each character to an indice.
- make tensor using this.
- adjust length of each word to be same.
- add start and stop words in every input word.    
- generate dataloader of data of batch size = batchSize(var)
    
### Encoder/Decoder/seq2seq class : 

**Encoder class**
- Encoder class for without attention
- forward() --> Encode the input sequence into a fixed-size representation.

**Decoder class**
- Decoder class for without attention
- forward() --> Generate the output sequence by decoding the encoded representation.

**Seq2Seq class**
- sequence class for without attention
- pl.LightningModule is used
- forward() --> Encode the source sequence, then decode it to generate the target sequence.
- Below mentioned functions are the functions of the lighteningModule.
- training_step() --> called at every training step.calculates the loss and accuracy of the model during training.We are logging it on wandb.
- validation_step() --> calculates the loss and accuracy of the model during validation.We are logging it on wandb.
- test_step() --> calculates the loss and accuracy of the model during testing.
- on_test_epoch_end() --> calculates loss and accuracy at the end of the test.


### AttnEncoder/AttnDecoder/Seq2SeqAttn class(with attention)
**AttnEncoder class**
- Encoder class for encoding with attention
- forward() --> Performs a forward pass of the Attention Encoder module.

**AttnDecoder class**
- Decoder class for decoding with attention
- forward() --> Performs a forward pass of the Attention Decoder module.

**Seq2SeqAttn class**
- sequence class for with attention
- pl.LightningModule is used
- forward() --> forward pass for seq2seqAttn. Encode the original sequence and then decode it to produce the target sequence.
- Below mentioned functions are the functions of the lighteningModule.
- training_step() --> called at every training step.calculates the loss and accuracy of the model during training.We are logging it on wandb.
- validation_step() --> calculates the loss and accuracy of the model during validation.We are logging it on wandb.
- test_step() --> calculates the loss and accuracy of the model during testing.
- on_test_epoch_end() --> calculates loss and accuracy at the end of the test.

**Init params with passed values**
- In this section we are initialising different parameters with the values passed as arguments. If the values are not passed then it will be initialised to default value(val for best cofig).

**Create model and Train+Test it**
- We are using pl.Trainer() , .fit() , .test() functions provided by PyTorch Lightning for training model on Train and validation data and evaluating on Test data.


## Additional functions in DL_Assignment2.ipynb the file
•	**_Sweep_**: 
The hyperparameters considered for sweep are:(for both the swee with and without attention)
-   Cell Type
-   Attention
-   Epoch
-   Drop Out
-   Learning rate
-   Embedding size
-   Hidden Layer Size
-   Encoder Layers
-   Decoder Layers
-   Bidirectional
•	**train()** -> This is train function for sweep for above parameters for with attention
•	**train_no_attn()** -> This is train function for sweep for above parameters for with no attention
