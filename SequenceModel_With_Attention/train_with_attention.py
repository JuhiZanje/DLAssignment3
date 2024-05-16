import wandb
import argparse
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import torch
import torch.nn.functional as F
torch.cuda.empty_cache()
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.utils.data as data
import pandas as pd
import os
import csv
warnings.filterwarnings("ignore")

wandb.login(key="494428cc53b5c21da594f4fc75035d136c63a93c")
wandb.init(project="CS6910_Assignment3")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#####################################----------------Set arguments-------------########################################################

arguments = argparse.ArgumentParser()
arguments.add_argument('-ct','--cell_type',type=str,default='LSTM')
arguments.add_argument('-at','--attention',type=bool,default=True)
arguments.add_argument('-e', '--epochs',  type=int, default=10)
arguments.add_argument('-do','--drop_out',type=float,default=0.5)
arguments.add_argument('-lr', '--learning_rate',type=float, default=0.001)
arguments.add_argument('-es','--embedding_size',type=int,default=128)
arguments.add_argument('-hs','--hidden_layer_size',type=int,default=256)
arguments.add_argument('-ne','--encoder_layers',type=int,default=3)
arguments.add_argument('-nd','--decoder_layers',type=int,default=3)
arguments.add_argument('-bd','--bidirectional',type=bool,default=True)
ter_args=arguments.parse_args()

#####################################----------------Data Preprocessing-------------########################################################
trainFilepath="/kaggle/input/aksharantar-sampled/aksharantar_sampled/hin/hin_train.csv"
valFilePath="/kaggle/input/aksharantar-sampled/aksharantar_sampled/hin/hin_valid.csv"
testFilePath="/kaggle/input/aksharantar-sampled/aksharantar_sampled/hin/hin_test.csv"
#Data Preprocessing
# Load the CSV file and retrieve the character sequence

def read_file_0(trainFilepath):
    with open(trainFilepath, 'r') as f:
        reader = csv.reader(f)
        chars = []
        for row in reader:
            chars.extend(row[0])  # assuming that the text data is in the first column of the CSV file
    return chars

'''Location of your CSV file (Extracted file)
Location of your CSV file if on kaggle than zip file is fine'''
trainFilepath = trainFilepath


chars = read_file_0(trainFilepath)
setChar=set(chars)
setChar.add('|')
setOfchar = list(setChar)

# Create the association between characters and their corresponding integer indices
char_to_idx_latin= {char: i+1 for i, char in enumerate(setOfchar)}

def read_file_1(trainFilepath):
    with open(trainFilepath, 'r') as f:
        reader = csv.reader(f)
        chars = []

        for r in reader:
            chars.extend(r[1])
    return chars

maxLenDev=0

chars = read_file_1(trainFilepath)
setChar=set(chars)
setChar.add('|')
setOfchar = list(setChar)

charToIndLang ={char: i+1 for i, char in enumerate(setOfchar)}

# Load the CSV file and retrieve the maxlen of word
with open(trainFilepath, 'r') as f:
    fileReader = csv.reader(f)
    chars = []

    wordLen = 0
    maxLenEng = 0
    fileIterator = iter(fileReader)

    while True:
        try:
            row = next(fileIterator)
            wordLen = len(row[0])
            if wordLen > maxLenEng:
                maxLenEng = wordLen
        except StopIteration:
            break

# Load the CSV file and retrieve the maxlen of word

with open(trainFilepath, 'r') as f:
    fileReader = csv.reader(f)
    chars = []

    wordLen = 0
    maxLenDev = 0

    while True:
        try:
            row = next(fileReader)
            wordLen = len(row[1])
            if wordLen > maxLenDev:
                maxLenDev = wordLen
        except StopIteration:
            break


#func to use char-ind to map char to ind
def convert_characters_to_indices(word, dictionary):
    indices = [dictionary.get(c, -1) for c in word]
    indices = [idx for idx in indices if idx >= 0]
    return indices

#function to make all words of same size
def adjust_sequence_length(indices, maximumLength):
    diff = maximumLength - len(indices)
    if diff < 0:
        indices = indices[:maximumLength]
    # If needed, add padding to ensure the sequence length equals maximumLength
    elif diff > 0:
        indices += [0] * (maximumLength - len(indices))
    return indices

#fun to covert indices to tensor
def convert_indices_to_tensor(indices, dictionary):
    start_token = dictionary.get('|', 0)
    end_token = dictionary.get('|', 0)
    indices = [start_token] + indices + [end_token]
    indTens = torch.tensor(indices)
    indTens = indTens.to(device)
    return indTens

#fun to covert word to indices
def convert_word_to_indices(word, maximumLength,dict):
    indices = convert_characters_to_indices(word, dict)
    indices = adjust_sequence_length(indices, maximumLength)
    indTens = convert_indices_to_tensor(indices, dict)
    return indTens

def assign_tensor_to_generated_sequences(sequence):
    seq_list = sequence.split()
    final_tensor = ""
    for word in seq_list:
        final_tensor+=word

    final_length = 0
    for word in seq_list:
        final_length += len(word)

    return final_tensor,final_length


def assemble_tensor(final_tensor,partition_size=1):
    if partition_size <= 0:
        partition_size = 1
    tensor_word_list = []
    for i in range(0,len(final_tensor),partition_size):
        tensor_word_list.append(final_tensor[i:i+partition_size])
    return tensor_word_list


def assemble_assigned_generated_seq(path):
    final_tensor,final_length = assign_tensor_to_generated_sequences(path)
    tensor_word_list = assemble_tensor(final_tensor,(int)(final_length/4))
    return tensor_word_list


def generate_indices(row):
    latin_word = row[0]
    devanagari_word = row[1]
    engInd = convert_word_to_indices(latin_word, maxLenEng,char_to_idx_latin)
    hindInd= convert_word_to_indices(devanagari_word,maxLenDev ,charToIndLang)
    return engInd,hindInd

#processing validation file
pairs_v=[]
with open(valFilePath, 'r') as f_v:
    reader_v = csv.reader(f_v)
    for row in reader_v:
        engInd,hindInd = generate_indices(row)
        pairs_v.append([engInd,hindInd])

#processing test file
pairs_t=[]
with open(testFilePath, 'r') as f_t:
    reader_t = csv.reader(f_t)
    for row in reader_t:
        engInd,hindInd = generate_indices(row)
        pairs_t.append([engInd,hindInd])

#processing train file
pairs=[]
with open(trainFilepath, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        engInd,hindInd = generate_indices(row)
        pairs.append([engInd,hindInd])                    


#get data loaders
batchSize=32
shuffleValTest=False
shuffleTrain=True
dataloaderVal = torch.utils.data.DataLoader(pairs_v, batch_size=batchSize, shuffle=shuffleValTest)
dataloaderTest = torch.utils.data.DataLoader(pairs_t, batch_size=batchSize, shuffle=shuffleValTest)
dataloaderTrain = torch.utils.data.DataLoader(pairs, batch_size=batchSize, shuffle=shuffleTrain)

########################----------------AttnEncoder/AttnDecoder/Seq2SeqAttn class(with attention)-------------####################################################

#AttnEncoder class for with attention

class AttnEncoder(nn.Module):
  #initialise the Attnencoder class with given params
    def __init__(self, inpDim, hiddenDim, embeddingSize,cellType,drop_out,num_layers,bidirectional):
        super().__init__()
        self.hiddenDim = hiddenDim
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(inpDim, embeddingSize)

        if(cellType=="GRU"):
          self.rnn = nn.GRU(embeddingSize, hiddenDim,dropout=drop_out,num_layers=num_layers,bidirectional=bidirectional)
        if(cellType=="LSTM"):
          self.rnn = nn.LSTM(embeddingSize, hiddenDim,dropout=drop_out,num_layers=num_layers,bidirectional=bidirectional)
        if(cellType=="RNN"):
          self.rnn = nn.RNN(embeddingSize, hiddenDim,dropout=drop_out,num_layers=num_layers,bidirectional=bidirectional)

    #performs forward pass on the AttnEncoder.
    def forward(self, x):
        output, hidden = self.rnn(self.embedding(x))
        return output, hidden

#AttnDecoder class for without attention
class AttnDecoder(nn.Module):
    #initialise AttnDecoder with given params.
    def __init__(self, opDim, hiddenDim,embeddingSize ,cellType,drop_out,num_layers,bidirectional,maximumLength):

        self.maximumLength=maximumLength+2
        super(AttnDecoder, self).__init__()
        self.embedding = nn.Embedding(opDim, embeddingSize)
        self.hiddenDim = hiddenDim
        self.bidirectional = bidirectional
        varToLinerArg=(self.hiddenDim + embeddingSize)
        self.attn = nn.Linear(varToLinerArg, self.maximumLength)
        self.cellType=cellType
        dimention=1
        if self.bidirectional:
          dimention=2

        varToAttnCombineArg=(self.hiddenDim*dimention + embeddingSize)
        self.attn_combine = nn.Linear(varToAttnCombineArg, self.hiddenDim)

        varCellTypeGRU=(cellType=="GRU")
        varCellTypeLSTM=(cellType=="LSTM")
        varCellTypeRNN=(cellType=="RNN")

        if(varCellTypeGRU):
          self.rnn = nn.GRU(hiddenDim, hiddenDim,dropout=drop_out,num_layers=num_layers ,bidirectional=bidirectional)
        elif(varCellTypeRNN):
          self.rnn = nn.RNN(hiddenDim, hiddenDim,dropout=drop_out,num_layers=num_layers,bidirectional=bidirectional)
        elif(varCellTypeLSTM):
          self.rnn = nn.LSTM(hiddenDim, hiddenDim,dropout=drop_out,num_layers=num_layers,bidirectional=bidirectional)

        varToLiner2Arg=hiddenDim*dimention
        self.fc = nn.Linear(varToLiner2Arg, opDim)

    '''
    forward pass for the decoder.
    To decode the encoded representation and generate the output sequence
    we will also calculate Attention weights
    '''
    def forward(self, x, hidden, encoderOutput):
        x = x.unsqueeze(1)
        x=x.transpose(0,1)

        varTocellType= (self.cellType=="LSTM")
        embedded = self.embedding(x)
        if(varTocellType):
          selfAttn=self.attn(torch.cat((embedded[0], hidden[0][0]), 1))
          attentionWeights = F.softmax(selfAttn, dim=1)
        else:
          selfAttn=self.attn(torch.cat((embedded[0], hidden[0]), 1))
          attentionWeights = F.softmax(selfAttn, dim=1)

        #resize attentionWeights and  encoderOutput
        attentionApplied = torch.bmm(attentionWeights.unsqueeze(1),
                                 encoderOutput.permute(1,0,2))
        output = torch.cat((embedded[0], attentionApplied.squeeze(1)), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        varTocellType= (self.cellType=="LSTM")
        #as LSTM return 2 things hidden and cell
        if(varTocellType):
          (hidden,cell)=hidden

        if(varTocellType):
          output, hidden = self.rnn(output, (hidden,cell))
        else:
          output, hidden = self.rnn(output, hidden)

        prediction = self.fc(output.squeeze(0))
        return prediction, hidden, attentionWeights

#Sequence class for with attention
class Seq2SeqAttn(pl.LightningModule):
    #initialise the Seq2SeqAttn class with given params
    def __init__(self, inpDim, opDim, hiddenDim,embeddingSize, cellType, drop_out,layersEncoder,layersDecoder,bidirectional,learningRate,maxLenEng):

        super().__init__()

        self.layersEncoder=layersEncoder
        self.learningRate=learningRate
        self.layersDecoder=layersDecoder
        self.cellType=cellType

        self.valLoss=[]
        self.valAccuracy=[]

        self.trainLoss=[]
        self.trainAccuracy=[]

        self.test_loss=[]
        self.testAccuracy=[]
        self.bidirectional = bidirectional

        self.encoder = AttnEncoder(inpDim, hiddenDim, embeddingSize,cellType,drop_out,layersEncoder,bidirectional)
        self.decoder = AttnDecoder(opDim, hiddenDim, embeddingSize, cellType,drop_out,layersDecoder,bidirectional,maxLenEng)
        self.dimention=1
        varTobidir=self.bidirectional
        if varTobidir:
          self.dimention=2

        self.counter=0
        self.attentionWeights=[]

    #forward pass for seq2seqAttn. Encode the original sequence and then decode it to produce the target sequence.
    #return output and attention weights
    def forward(self, src, trg,teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        max_len = trg.shape[1]
        src = src.transpose(0,1)
        attentionV = torch.zeros(max_len, batch_size, maxLenEng+2).to(self.device)

        vocabSizeTrgt = self.decoder.fc.out_features
        encoder_output,hidden = self.encoder(src)
        outputs = torch.zeros(max_len, batch_size, vocabSizeTrgt).to(self.device)

        input = trg[:,0]
        t = 1
        while t < max_len:
            output, hidden, attentionV[t] = self.decoder(input, hidden, encoder_output)
            outputs[t] = output
            if teacher_forcing_ratio < torch.rand(1).item():
                input = output.argmax(1)
            else:
                input = trg[:, t]
            t += 1
        return outputs, attentionV

    #fun to get model output
    def get_model_output(self, src, trg):
        output, attentionV = self(src, trg)
        output_acc, _ = self(src, trg)
        return output, output_acc.permute(1, 0, 2)

    #fun will return expected values tensor
    def prepare_expected_tensor(self, output, trg):
        rows = torch.arange(output.shape[0])
        cols = torch.arange(output.shape[1]).unsqueeze(1)
        expected = torch.zeros(size=output.shape)
        expected[rows, cols, trg.cpu()] = 1
        return expected

    def prepare_output_expected_tensors(self, output, trg):
        opDim = output.shape[-1]
        output = output[1:].view(-1, opDim)
        expected = self.prepare_expected_tensor(output, trg)[1:].view(-1, opDim)
        trg = trg[1:].view(-1)
        return output, expected, trg

    #helper func to find loss
    def calculate_loss(self, output, expected):
        return self.loss_fn(output.to(device), expected.to(device))

    #helper fun to find accuracy
    def calculate_accuracy(self, output, trg):
        return self.accuracy(output, trg)

    def append_metrics(self, trainLoss, trainAccuracy):
        self.trainAccuracy.append(torch.tensor(trainAccuracy))
        self.trainLoss.append(torch.tensor(trainLoss))

    #This fun will be called at every training step. return the loss.
    def training_step(self, batch, batch_idx):
        src, trg = batch

        trg_accuracy = trg
        output, output_acc = self.get_model_output(src, trg)

        output, expected, trg = self.prepare_output_expected_tensors(output, trg)

        trainAccuracy = self.calculate_accuracy(output_acc, trg_accuracy)

        trainLoss = self.calculate_loss(output, expected)

        self.append_metrics(trainLoss, trainAccuracy)

        return {'loss': trainLoss}


    def get_output(self, src, trg):
        output, attentionV = self(src, trg, 0)
        output_acc, attentionV = self(src, trg, 0)
        return output, output_acc.permute(1, 0, 2)

    def prepare_output_expected_tensors(self, output, trg):
        rows = torch.arange(output.shape[0])
        cols = torch.arange(output.shape[1]).unsqueeze(1)
        expected = torch.zeros(size=output.shape)
        path = '/kaggle/input/aksharantar-sampled-5/aksharantar_sampled/hin/hin_val.csv'
        expected[rows, cols, trg.cpu()] = 1
        opDim = output.shape[-1]
        return output[1:].view(-1, opDim), expected[1:].view(-1, opDim), trg[1:].view(-1)

    def calculate_loss(self, output, expected):
        return self.loss_fn(output.to(device), expected.to(device))

    def calculate_accuracy(self, output, trg):
        return self.accuracy(output, trg)

    def assemble_and_save_sequences(self, path):
        assemble_assigned_generated_seq(path)

    #Operates on a single batch of data from the validation set.Return validationLoss.
    def validation_step(self, batch, batch_idx):
        src, trg = batch
        trg_accuracy = trg

        output, output_accuracy = self.get_output(src, trg)

        output, expected, trg = self.prepare_output_expected_tensors(output, trg)

        valLoss = self.calculate_loss(output, expected)

        valAccuracy = self.calculate_accuracy(output_accuracy, trg_accuracy)

        path = '/kaggle/input/aksharantar-sampled-5/aksharantar_sampled/hin/hin_val.csv'
        self.assemble_and_save_sequences(path)

        self.valAccuracy.append(torch.tensor(valAccuracy))
        self.valLoss.append(torch.tensor(valLoss))

        return {'loss': valLoss}


    #func to config optimizer
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learningRate )
        return optimizer

    #func to config loss fun
    def loss_fn(self, output, trg):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, trg)
        return loss.mean()

    #function to find accuracy given ouput and true value (word wise)
    def accuracy(self, output, target):
      predicted = output.argmax(dim=-1)
      equal_rows = 0
      i = 0
      while i < target.size(0):
          if torch.all(torch.eq(target[i, 1:-1], predicted[i, 1:-1])):
              equal_rows += 1
          i += 1
      matches=equal_rows

      accuracy = matches / len(target) * 100
      return accuracy

    #fun to create grid given input word,ouput word and predicted word
    def grid(self,input, output, target):
      inputGrid=[]
      trgGrid=[]
      expectedGrid=[]
      predicted = output.argmax(dim=-1)
      i = 0
      while i < target.size(0):
          trgGrid.append(target[i, 1:-1])
          expectedGrid.append(predicted[i, 1:-1])
          inputGrid.append(input[i, 1:-1])
          i += 1
      return inputGrid,trgGrid,expectedGrid

    '''
      Train Epoch-level Operations.
      Fun will be called after every epoch.
    '''
    def on_train_epoch_end(self):
      trainLoss=torch.stack(self.trainLoss).mean()
      self.trainLoss=[]

      valLoss=torch.stack(self.valLoss).mean()
      self.valLoss=[]

      trainAccuracy=torch.stack(self.trainAccuracy).mean()
      self.trainAccuracy=[]

      valAccuracy=torch.stack(self.valAccuracy).mean()
      self.valAccuracy=[]
      print({"Train Loss":trainLoss,"Train Accuracy":trainAccuracy,"Validation Loss":valLoss,"Validation Accuracy":valAccuracy})
      wandb.log({"Train Loss":trainLoss,"Train Accuracy":trainAccuracy,"Validation Loss":valLoss,"Validation Accuracy":valAccuracy})


    def get_out_attention(self,src,trg):
        output, attentionV = self(src, trg,0)
        outputAcc, attentionVD = self(src, trg,0)
        return output,attentionV,outputAcc.permute(1,0,2)

    def get_expected(self,output, trg):
        cols = torch.arange(output.shape[1]).unsqueeze(1)
        rows = torch.arange(output.shape[0])
        expected = torch.zeros(size=output.shape)
        expected[rows, cols, trg.cpu()] = 1
        return expected

    '''
      Operates on a single batch of data from the test set.
      When the test_step() is called, the model has been put in eval mode and PyTorch gradients have been disabled
    '''
    def test_step(self, batch, batch_idx):
        test_path = '/kaggle/input/aksharantar-sampled/aksharantar_sampled/hin/hin_val.csv'
        assemble_assigned_generated_seq(test_path)
        src, trg = batch
        trgAcc=trg

        output, attentionV, outputAcc = self.get_out_attention(src, trg)

        expected = self.get_expected(output, trg)
        opDim = output.shape[-1]

        output = output[1:].view(-1, opDim)
        expected = expected[1:].view(-1, opDim)
        trg = trg[1:].view(-1)

        test_loss = self.loss_fn(output.to(device), expected.to(device))
        testAccuracy =self.accuracy(outputAcc, trgAcc)
        inputGrid,trgGrid,grid_predicted=self.grid(src,outputAcc, trgAcc)

        assemble_assigned_generated_seq("string representations")
        #convert grid representation to string
        #for target string
        target_outputs=[]
        targetString=""
        for i in trgGrid:
          for j in i:
            integer_value = j.item()
            targetString=targetString+keyForVal(j)
          target_outputs.append(targetString)
          str_cp = (targetString + '.')[:-1]
          assemble_assigned_generated_seq(str_cp)
          targetString=""

        #for predicted string
        predicted_outputs=[]
        str_predicted=""
        for i in grid_predicted:
          for j in i:
            integer_value = j.item()
            str_predicted=str_predicted+get_keyAttn(j)
          predicted_outputs.append(str_predicted)
          str_cp = (str_predicted + '.')[:-1]
          assemble_assigned_generated_seq(str_cp)
          str_predicted=""

        #for input string
        inputs=[]
        inpString=""
        for i in inputGrid:
          for j in i:
            integer_value = j.item()
            inpString=inpString+keyForInput(j)
          inputs.append(inpString)
          inpString=""

        str_cp = (inpString + '.')[:-1]
        assemble_assigned_generated_seq(str_cp)
        self.testAccuracy.append(torch.tensor(testAccuracy))
        self.test_loss.append(torch.tensor(test_loss))
        # print({"for batch test_loss":test_loss,"testAccuracy":testAccuracy})
        wandb.log({"Test Loss":test_loss,"Test Accuracy":testAccuracy})
        # Save target and predicted outputs to a CSV file
        save_outputs_to_csvAttn(inputs,target_outputs, predicted_outputs)
        # plot_attention_weights(self.attentionWeights)
        if(self.counter<1):
          s(inputs,predicted_outputs,attentionV)
          self.counter=self.counter+1
        return {'loss':test_loss}

    def on_test_epoch_end(self):
        testAccuracy=torch.stack(self.testAccuracy).mean()
        self.testAccuracy=[]

        test_loss=torch.stack(self.test_loss).mean()
        self.test_loss=[]
        print({"test_loss":test_loss,"testAccuracy":testAccuracy})
        wandb.log({"test_loss_last":test_loss,"testAccuracy_last":testAccuracy})

        # wandb.log({"Train Loss":trainLoss,"Train Accuracy":trainAccuracy,"Validation Loss":valLoss,"Validation Accuracy":valAccuracy})


# function to return key for any value
def get_keyAttn(val):
    keys = list(charToIndLang.keys())
    values = list(charToIndLang.values())
    index = 0
    key = ""

    while index < len(values):
        if val == values[index]:
            key = keys[index]
            break
        index += 1

    return key


def get_key_inputAttn(val):
    keys = list(char_to_idx_latin.keys())
    values = list(char_to_idx_latin.values())
    index = 0
    key = ""

    while index < len(values):
        if val == values[index]:
            key = keys[index]
            break
        index += 1

    return key

#function will save the input word ,ouput word and predicted word to csv file for attention module
def save_outputs_to_csvAttn(inputs,target_outputs, predicted_outputs):
    file_exists = os.path.exists('Output_Attn.csv')
    dict = {'input':inputs,'target':target_outputs, 'predicted': predicted_outputs}
    df = pd.DataFrame(dict)
    df.to_csv('Output.csv',mode='a',index=False,header=not file_exists)

#function to create 3*3 grid of heatMap
def s(input_words, output_words, attentionWeights):
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, ax in enumerate(axes.flat):
        # Check if input and output words are provided
        if i < len(input_words) and i < len(output_words):
            # Get the attention weights for the corresponding input word
            attn_weight = attentionWeights[i].cpu().detach().numpy()
            attn_weight = attn_weight[1:len(input_words[i]) + 1, :len(output_words[i])]

            # Plot the attention weights as a heatmap on the current axis
            sns.heatmap(attn_weight, ax=ax, cmap='Blues', cbar=False)

            # Set the y-axis tick positions and labels to the input words and rotate them vertically
            ax.set_yticks(range(len(input_words[i])))
            ax.set_yticklabels(reversed(input_words[i]), rotation='vertical')

            # Set the x-axis tick positions and labels to the output words and rotate them horizontally
            ax.set_xticks(range(len(output_words[i])))
#             fontproperties=hindi_font,
            ax.set_xticklabels(reversed(output_words[i]), rotation=45, ha='right')

            # Set the title of each subplot as the index number
            ax.set_title(f'Attention {i+1}', fontsize=12)
        else:
            # If input or output words are missing, display a message in the subplot
            ax.text(0.5, 0.5, 'Missing Data', horizontalalignment='center', verticalalignment='center', fontsize=12)
            ax.axis('off')

    for j in range(len(input_words), len(axes.flat)):
        fig.delaxes(axes.flat[j])

    wandb.log({"Question 5": wandb.Image(plt)})

    plt.show()


#################################----------------init params with passed values-------------########################################################

attention=True
hidden_layer_size=ter_args.hidden_layer_size
embeddingSize= ter_args.embedding_size
cellType=ter_args.cell_type
layersDecoder=ter_args.decoder_layers
layersEncoder=ter_args.encoder_layers
bidirectional=ter_args.bidirectional
epochs=ter_args.epochs
learningRate=ter_args.learning_rate
drop_out=ter_args.drop_out


#################################----------------create model and Train+Test it-------------########################################################

#create model
if(attention==False):
  model = Seq2Seq(len(char_to_idx_latin)+2, len(charToIndLang)+2, hidden_layer_size, embeddingSize, cellType,drop_out,layersEncoder,layersDecoder,bidirectional,learningRate)

else:
  model = Seq2SeqAttn(len(char_to_idx_latin)+2, len(charToIndLang)+2, hidden_layer_size, embeddingSize, cellType,drop_out,1,1,bidirectional,learningRate, maxLenEng)

model.to(device)

#train and test model
# wandb.init()
trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu", devices=1)
trainer.fit(model=model, train_dataloaders=dataloaderTrain, val_dataloaders=dataloaderVal)
trainer.test(model, dataloaderTest)
wandb.finish()

