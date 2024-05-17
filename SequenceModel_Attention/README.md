# 1) DL_Assignment3_Attn.ipynb and train_attention.py


## DL_Assignment3_Attn.ipynb
- This file contains The code for all the question 5 given in assignment.
-   **Before running ensure to change datapaths in variable:(give path to train.csv,val.csv,test.csv)**
    - trainFilepath --> path to hin_train.csv
    - valFilePath   --> path to hin_valid.csv
    - testFilePath  --> path to hin_test.csv
- This is ipynb file.
- Thus run each cell one by one to see the output.
- It contains code for both seq2seq model (with attention and without attention)

## train_attention.py
-   This is a python file
-   You can set parameters value of your choice .
-   You can give arguments as well.
-   **Before running ensure to change datapaths in variable:(give path to train.csv,val.csv,test.csv)**
    - trainFilepath --> path to hin_train.csv
    - valFilePath   --> path to hin_valid.csv
    - testFilePath  --> path to hin_test.csv
-   I have run the file on kaggle thus data paths given for dataset are in that way.(You need to change it accordingly)
-   train() and test() methods are called at the end of the file thus the model will be trained and tested and the accuracies will be printed as well as a sweep run will be created.

### If you want to Run train_attention.py on kaggle*
-  First upload dataset on kaggle.
-  then upload train.py file as well  
-  run code -> !pip install argparse 
-   Before running ensure to change datapaths in variable:(give path to train.csv,val.csv,test.csv)
    - trainFilepath
    - valFilePath
    - testFilePath
-  To run now -> !python $path_to_uploaded_train_attention.py file(on kaggle)
-   eg : !python /kaggle/input/train-attn/train_attention.py
-  You can give given below args while running file
-   eg : !python /kaggle/input/train-attn/train_attention.py -e 1 -ct "RNN"


### If you want to Run train_attention.py on CMD*
-  When needed Will have to install some packages.
-   Before running ensure to change datapaths in variable:(give path to train.csv,val.csv,test.csv)
    - trainFilepath
    - valFilePath
    - testFilePath 
-   after this just run -->  python train_attention.py 
-   You can give given below args while running file

## Parameter choices which can be pass as command line arguments
-   Cell Type : '-ct','--cell_type', choices=["RNN", "GRU", "LSTM"]
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


## Additional functions in DL_Assignment3_Attn.ipynb the file
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
