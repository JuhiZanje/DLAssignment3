# CS6910 Assignment 3
The provided code snippet demonstrates how to implement a sequence-to-sequence (Seq2Seq) model using PyTorch. It begins by preparing and preprocessing a dataset consisting of English-Hindi word pairs. The code defines both an Encoder and a Decoder class, utilizing RNN cells such as LSTM, GRU, or vanilla RNN. The Seq2Seq model is then initialized with these components. Finally, the forward pass is executed, where the source sequence is encoded, and the target sequence is generated through decoding.

## SequenceModel_Attention (Folder)
- Which contains code for SequenceModel_With_Attention
- This Folder Contains files :
    - train_attention.py
    - DL_Assignment3_Attn.ipynb
    - README.md
    - predictions_attention (Folder) <--For Question 5(b)
        -   Output_Attn.csv
- This folder has its own readme file.

## SequenceModel_Vanila (Folder)
- Which contains code for SequenceModel_Without_Attention
- This Folder Contains files :
    - train_vanila.py
    - DL_Assignment3_vanila.ipynb
    - README.md
    - predictions_vanila (Folder)
        -   Output.csv
- This folder has its own readme file.

### I have kept other files here As I first pushed them here . So those file contains my early commits.
like --> 
-   DL_ASSIGNMENT3.ipynb -> contains code for everything