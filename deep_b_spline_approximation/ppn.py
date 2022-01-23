# -*- coding: utf-8 -*-
"""
Created on Thu May 14 22:45:04 2020

@author: Tommaso
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from deep_b_spline_approximation.BSpline import NDPWithBatch2,computeControlPointsWithBatch,computeControlPointsWithBatch2
from timeit import default_timer as timer

torch.set_default_dtype(torch.float64)
 
"""PPN-MLP"""

class PointParametrizationNetwork(nn.Module):
    
    def __init__(self,inputSize,hiddenSize,p,k=3,device="cuda"):
        super(PointParametrizationNetwork,self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.p = p
        self.l = int(self.inputSize/self.p)
        self.outputSize = self.l - 1
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
          
        self.k = k
        self.u = torch.cat((torch.tensor([0.0]*(self.k+1)),torch.tensor([1.0]*(self.k+1))))
        
        #Fully connected layers
        self.inputLayer = torch.nn.Linear(self.inputSize,self.hiddenSize)
        self.hiddenLayer1 = torch.nn.Linear(self.hiddenSize,self.hiddenSize)
        self.hiddenLayer2 = torch.nn.Linear(self.hiddenSize,self.hiddenSize)
        self.hiddenLayer3 = torch.nn.Linear(self.hiddenSize,self.hiddenSize)
        self.outputLayer = torch.nn.Linear(self.hiddenSize,self.outputSize)
        
        self.softplus = torch.nn.Softplus()
        #self.relu = torch.nn.ReLU()
        
        self.batchNormInput = torch.nn.BatchNorm1d(self.hiddenSize,momentum=0.1,track_running_stats=False)
        self.batchNorm1 = torch.nn.BatchNorm1d(self.hiddenSize,momentum=0.1,track_running_stats=False)
        self.batchNorm2 = torch.nn.BatchNorm1d(self.hiddenSize,momentum=0.1,track_running_stats=False)
        self.batchNorm3 = torch.nn.BatchNorm1d(self.hiddenSize,momentum=0.1,track_running_stats=False)
        self.batchNormOutput = torch.nn.BatchNorm1d(self.outputSize,momentum=0.1,track_running_stats=False)
        
        self.layerNormInput = torch.nn.LayerNorm(self.hiddenSize)
        self.layerNorm1 = torch.nn.LayerNorm(self.hiddenSize)
        self.layerNorm2 = torch.nn.LayerNorm(self.hiddenSize)
        self.layerNorm3 = torch.nn.LayerNorm(self.hiddenSize)
        self.layerNormOutput = torch.nn.LayerNorm(self.outputSize)
        
    
    def forward(self,x):
        batchSize = x.shape[0]
        
        curves = torch.zeros((batchSize,self.l,self.p))
        curves = x.reshape(batchSize,self.p,self.l).permute(0,2,1).to(self.device)
        
        x = self.inputLayer(x)
        x = self.batchNormInput(x)
        x = self.softplus(x)
        
        x = self.hiddenLayer1(x)
        x = self.batchNorm1(x)
        x = self.softplus(x)
        
        x = self.hiddenLayer2(x)
        x = self.batchNorm2(x)
        x = self.softplus(x)
        
        x = self.hiddenLayer3(x)
        x = self.batchNorm3(x)
        x = self.softplus(x)
        
        x = self.outputLayer(x)
        x = self.batchNormOutput(x)
        x = self.softplus(x)
        
        """PP1. Accumulation & Rescaling Layer"""
        
        x = torch.cumsum(x,1)
        x = torch.cat((torch.zeros(x.shape[0],1).to(self.device),x),axis=1)
        
        xmax = torch.max(x,1)[0].view(-1,1)
        x = torch.div(x,xmax)
        
        """PP2. Approximation Layer"""
        
        A = NDPWithBatch2(x,self.k,self.u,self.device)
        
        c = torch.zeros((batchSize,A.shape[2],self.p)).to(self.device)
        splines = torch.zeros((batchSize,self.l,self.p)).to(self.device)
       
        c = computeControlPointsWithBatch2(A,curves)
        
        splines = torch.matmul(A,c)
        
        return splines,x,curves

"""PPN-CNN"""

class PointParametrizationNetworkCNN2(nn.Module):
    def __init__(self, 
                 inputDim=[2,200,200,500],
                 outputDim=1,
                 hiddenDim=[200,200,500,500],
                 nLayers=4,
                 kernelSize=[11,11,7,3],
                 fcDimIn = [4500,3000,1500,1000,500],
                 fcDimOut = [3000,1500,1000,500,100],
                 k=3,
                 seqLen=100,
                 device="cuda:0"):
        super().__init__()
        
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.nLayers = nLayers
        self.outputDim = outputDim
        
        self.k = k
        self.u = torch.cat((torch.tensor([0.0]*(self.k+1)),torch.tensor([1.0]*(self.k+1))))
        
        self.device = device
        
        self.convs = nn.ModuleList([nn.Conv1d(in_channels = inputDim[i], 
                                              out_channels = hiddenDim[i], 
                                              kernel_size = kernelSize[i])
                                    for i in range(nLayers)])
        
        
        self.bnconvs = nn.ModuleList([nn.BatchNorm1d(hiddenDim[i],
                                                     momentum=0.1,
                                                     track_running_stats=False) 
                                     for i in range(nLayers)])
        
        
        self.fcs = nn.ModuleList([torch.nn.Linear(fcDimIn[i],fcDimOut[i]) for i in range(5)])
        
        self.bnfcs = nn.ModuleList([nn.BatchNorm1d(fcDimOut[i],
                                                     momentum=0.1,
                                                     track_running_stats=False) 
                                     for i in range(5)])
        
        self.relu = torch.nn.ReLU()
        self.softplus = torch.nn.Softplus()
        
        self.dropout = nn.Dropout(0.2)
        
       
    def forward(self,x):
        batchSize = x.shape[0]
        l = x.shape[1]
        p = x.shape[2]
        
        curves = x.clone()
        
        convInput = x.permute(0,2,1)
        
        for i,(conv,bn) in enumerate(zip(self.convs,self.bnconvs)):
       
            conved = conv(convInput)
            
            conved = bn(conved)
            
            conved = self.relu(conved)
            
        
            if i < 2:
                conved = F.max_pool1d(conved,2)
            
            convInput = conved
        
        
        linearInput = conved.view(-1,conved.shape[1]*conved.shape[2])
        
        for i,(fc,bn) in enumerate(zip(self.fcs,self.bnfcs)):
            
            linear = fc(linearInput)
            linear = bn(linear)
            linear = self.softplus(linear)
            
            linearInput = linear
        
        linear = linear.squeeze()
        
        #PP1. Accumulation & Rescaling Layer
        
        t = torch.cumsum(linear,1)
        tmin = torch.min(t,1)[0].view(-1,1)
        tmax = torch.max(t,1)[0].view(-1,1)
        t = torch.div(t-tmin,tmax-tmin)
        
        if len(t.unique()) < len(t):
            print("t si ripete")
        
        #PP2. Approximation Layer
        
        A = NDPWithBatch2(t,self.k,self.u,self.device)
        
        c = torch.zeros((batchSize,A.shape[2],p)).to(self.device)
        splines = torch.zeros((batchSize,l,p)).to(self.device)
        
        c = computeControlPointsWithBatch2(A,curves)
        
        splines = torch.matmul(A,c)
        
        return splines,t,curves

"""PPN-RNN"""

class PointParametrizationNetworkLSTM(nn.Module):
    def __init__(self, inputDim=2,outputDim=1,hiddenDimEnc=512,hiddenDimDec=1,nLayers=2,hiddenLinear=256,seqLen = 100,k=3,device="cuda:0"):
        super().__init__()
        
        self.inputDim = inputDim
        self.hiddenDimEnc = hiddenDimEnc
        self.hiddenDimDec = hiddenDimDec
        self.hiddenLinear = hiddenLinear
        self.nLayers = nLayers
        self.outputDim = outputDim
        
        self.k = k
        self.u = torch.cat((torch.tensor([0.0]*(self.k+1)),torch.tensor([1.0]*(self.k+1))))
        
        self.device = device
        
        self.fc1 = torch.nn.Linear(self.inputDim,self.hiddenLinear)
        
        self.rnn1 = nn.LSTM(self.hiddenLinear,self.hiddenDimEnc,num_layers=nLayers,batch_first=True)
        #self.rnn2 = nn.LSTM(self.hiddenDimEnc,self.hiddenDimEnc,num_layers=nLayers,batch_first=True)
        #self.rnn3 = nn.LSTM(self.hiddenDimEnc,self.hiddenDimEnc,num_layers=nLayers,batch_first=True)
        
        self.fc2 = torch.nn.Linear(self.hiddenDimEnc,self.outputDim)
        #self.fc3 = torch.nn.Linear(self.hiddenLinear,self.hiddenLinear)
        #self.fc4 = torch.nn.Linear(self.hiddenLinear,self.outputDim)
        
        self.decoder = nn.LSTM(self.inputDim+self.hiddenDimEnc,self.hiddenDimDec,num_layers=nLayers,batch_first=True)
        
        self.layerNorm1 = torch.nn.LayerNorm(self.hiddenDimEnc)
        
        self.batchNorm1 = torch.nn.BatchNorm1d(seqLen,momentum=0.1,track_running_stats=False)
        #self.batchNorm2 = torch.nn.BatchNorm1d(self.hiddenDimEnc,momentum=0.1,track_running_stats=False)
        #self.batchNorm3 = torch.nn.BatchNorm1d(self.hiddenDimEnc,momentum=0.1,track_running_stats=False)
        
        self.relu = torch.nn.ReLU()
        self.softplus = torch.nn.Softplus()
        
       
    def forward(self,x):
        batchSize = x.shape[0]
        l = x.shape[1]
        p = x.shape[2]
        
        curves = x.clone()
        
        x = self.fc1(x)
        
        x = self.batchNorm1(x)
        
        (hidden, cell) = (torch.zeros(2,batchSize,self.hiddenDimEnc,device = self.device), 
                          torch.zeros(2,batchSize,self.hiddenDimEnc,device = self.device))
        
        output, (hidden, cell) = self.rnn1(x,(hidden, cell))
        
        x = self.fc2(self.layerNorm1(output[:,1:]))
        
        #x = self.batchNorm1(x)
        x = self.softplus(x)
        
        
        x = torch.squeeze(x)
       
        """PP1. Accumulation & Rescaling Layer"""
        
        #print(x.shape)
        
        x = torch.cumsum(x,1)
        #print(x.shape)
        x = torch.cat((torch.zeros(x.shape[0],1).to(self.device),x),axis=1)
        xmax = torch.max(x,1)[0].view(-1,1)
        #x = torch.div(x.T,xmax).T
        x = torch.div(x,xmax)
        
        #print(f"x dopo aver aggiunto 0 {x.shape}")
        
        if len(x.unique()) < len(x):
            print("x si ripete")
        
        
        """PP2. Approximation Layer"""
        
        A = NDPWithBatch2(x,self.k,self.u,self.device)
        
        c = torch.zeros((batchSize,A.shape[2],p)).to(self.device)
        splines = torch.zeros((batchSize,l,p)).to(self.device)
        
        #print(A.shape)
        #print(curves.shape)
        
        c = computeControlPointsWithBatch2(A[:,1:-1],curves[:,1:-1])
        
        splines = torch.matmul(A,c)
        
        return splines,x,curves

"""PPN-SEQ2SEQ-RNN"""

class EncoderRNN(nn.Module):
    
    def __init__(self,inputSize,hiddenSize,hiddenSizeLinear=256,seqLen=100,bidirectional=True,device="cuda:0"):
        
        super(EncoderRNN, self).__init__()
        self.hiddenSize = hiddenSize
        self.inputSize = inputSize
        self.hiddenSizeLinear = hiddenSizeLinear
        self.bidirectional = bidirectional
        
        self.fc1 = nn.Linear(inputSize,hiddenSizeLinear)
        
        self.bn1 = torch.nn.BatchNorm1d(seqLen,momentum=0.1,track_running_stats=False)
    
        self.lstm = nn.LSTM(hiddenSizeLinear,hiddenSize,bidirectional=bidirectional,batch_first=True)
        
        self.device = device
  
    def forward(self, inputs, hidden):
        batchSize = inputs.shape[0]
        l = inputs.shape[1]
        
        inputs = self.fc1(inputs)
        
        inputs = self.bn1(inputs)
    
        output, hidden = self.lstm(inputs.view(batchSize,l,self.hiddenSizeLinear), hidden)
    
        return output, hidden
    
    def initHidden(self,batchSize):
        
        return (torch.zeros(1 + int(self.bidirectional),batchSize,self.hiddenSize).to(self.device),
                torch.zeros(1 + int(self.bidirectional),batchSize,self.hiddenSize).to(self.device))
    
class AttentionDecoderRNN(nn.Module):
    
    def __init__(self, inputSize, hiddenSizeEnc, hiddenSizeDec, finalSize, hiddenSizeLinear=256, seqLen=100,bidirectional=True):
        
        super(AttentionDecoderRNN, self).__init__()
        
        self.inputSize = inputSize
        self.hiddenSizeEnc = hiddenSizeEnc
        self.bidirectional = bidirectional
        self.hiddenSizeDec = (1 + int(self.bidirectional))*hiddenSizeEnc
        
        self.W = nn.Linear((1 + int(self.bidirectional))*hiddenSizeEnc,seqLen)
        self.V = nn.Linear((1 + int(self.bidirectional))*hiddenSizeEnc,seqLen)
        self.v = nn.Linear(seqLen,1)
        
        self.fc1 = nn.Linear(inputSize,(1 + int(self.bidirectional))*hiddenSizeEnc)
        
        self.lstm = nn.LSTM(self.hiddenSizeDec*2,self.hiddenSizeEnc,bidirectional=bidirectional,batch_first=True)            #inputSize = 21, ossia hidden+t[i]
        
        self.fc2 = nn.Linear(2*self.hiddenSizeEnc,self.hiddenSizeEnc)
        self.fc3 = nn.Linear(self.hiddenSizeEnc,finalSize)
        
        self.bn2 = torch.nn.BatchNorm1d(seqLen,momentum=0.1,track_running_stats=False)
        
    def forward(self, decoderHidden, encoderOutputs, input):
        
        batchSize = encoderOutputs.shape[0]
        
        #Reshape decoder hidden for the computation of attention 
        #encoderOutputs = encoderOutputs.permute(0,2,1)
        
        hiddenState = decoderHidden[0].permute(1,0,2)
        #hiddenState = hiddenState.permute(0,2,1)
        hiddenState = hiddenState.reshape(batchSize,1,-1)
        
        #Define attention
        
        u = torch.tanh(self.W(encoderOutputs) + self.V(hiddenState))
        u = self.v(u)
        
        attention = F.softmax(u,1)
        
        #print(f"attention vector {attention.shape}")
        
        encoderOutputs = encoderOutputs.permute(0,2,1)          #reshape encoder outputs for application of attention
        context = torch.bmm(encoderOutputs,attention)           #apply attention for computation of context
        
        #Prepare the input for the decoder
        
        input = input.view(-1,1,1)
        input = self.fc1(input)
        input = input.permute(0,2,1)
        
        decoderInput = torch.cat((input,context),dim=1)
        decoderInput = decoderInput.permute(0,2,1)
        
        output, hidden = self.lstm(decoderInput,decoderHidden)
        
        output = output.squeeze()
        
        output = self.fc2(output)
        
        output = self.fc3(output)
        
        return output,hidden,attention
    
    def initHidden(self,batchSize):
        
        return (torch.zeros(1,batchSize, self.hiddenSize).to(self.device),
                torch.zeros(1,batchSize, self.hiddenSize).to(self.device))



class Seq2SeqRNN(nn.Module):
    
    def __init__(self,inputSizeEncoder,hiddenSizeEncoder,bidirectional,inputSizeDecoder,hiddenSizeDecoder,finalSize,k=3,device="cuda:0"):
        
        super(Seq2SeqRNN,self).__init__()
        
        self.bidirectional = bidirectional
        #self.hiddenSizeEncoder = hiddenSizeEncoder
        self.hiddenSizeEncoder = hiddenSizeEncoder
        
        self.encoder = EncoderRNN(inputSizeEncoder,hiddenSizeEncoder,bidirectional=self.bidirectional)
        self.decoder = AttentionDecoderRNN(inputSizeDecoder,hiddenSizeEncoder,hiddenSizeDecoder,finalSize)
        
        self.k = k
        self.u = torch.cat((torch.tensor([0.0]*(self.k+1)),torch.tensor([1.0]*(self.k+1))))
        
        self.device = device
        
            
    def initHidden(self,batchSize):
        
        return (torch.zeros(1 + int(self.bidirectional),batchSize,self.hiddenSizeEncoder),
                torch.zeros(1 + int(self.bidirectional),batchSize,self.hiddenSizeEncoder))
        
    
    def forward(self,input):
        
        batchSize = input.shape[0]
        l = input.shape[1]
        p = input.shape[2]
        curves = input.clone()
        
        encoderOutputs, encoderHidden = self.encoder(input,self.encoder.initHidden(batchSize))
        #decoderHidden = encoderHidden[0][0]
        decoderHidden = encoderHidden
        
        decoderInput = torch.zeros((batchSize,1)).to(self.device)
        decoderOutputs = torch.zeros((batchSize,l,1)).to(self.device)
        decoderOutputs[:,0] = decoderInput
        
        attentionMatrix = torch.zeros((l,l))                                      # for visualization
            
        for j in range(1,l):
            
            #print(f"punto n. {j}")
                
            decoderOutput,decoderHidden,decoderAttention = self.decoder(decoderHidden,encoderOutputs,decoderInput)
            
            decoderOutputs[:,j] = F.softplus(decoderOutput)
                
            decoderInput = F.softplus(decoderOutput)
            
            decoderAttention = torch.mean(decoderAttention,0)
            
            decoderAttention = decoderAttention.permute(1,0)
            
            attentionMatrix[j] = decoderAttention
        
        
        decoderOutputs = decoderOutputs.squeeze(-1)
        
        #PP1. Accumulation & Rescaling Layer
        
        t = torch.cumsum(decoderOutputs,1)
        #t = torch.cat((torch.zeros(t.shape[0],1).to(self.device),t),axis=1)
        tmax = torch.max(t,1)[0].view(-1,1)
        t = torch.div(t,tmax)
        
        if len(t.unique()) < len(t):
            print("t si ripete")
        
        
        #PP2. Approximation Layer
        
        A = NDPWithBatch2(t,self.k,self.u,self.device)
        
        c = torch.zeros((batchSize,A.shape[2],p)).to(self.device)
        splines = torch.zeros((batchSize,l,p)).to(self.device)
        
        
        c = computeControlPointsWithBatch2(A[:,1:-1],curves[:,1:-1])
        
        splines = torch.matmul(A,c)
        
        return splines,t,curves


"""PPN-SEQ2SEQ-CNN"""

class Seq2SeqCNN(nn.Module):
    def __init__(self,encoder,decoder):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self,curves):
        
        encoderConved, encoderCombined = self.encoder(curves)
            
        splines,t,curves,attention = self.decoder(curves[:,:-1],encoderConved,encoderCombined)
        
        return splines,t,curves

class EncoderCNN(nn.Module):
    def __init__(self,inputDim,embDim,hiddenDim,nLayers,kernelSize,dropout,device,maxLenght = 100):
        super().__init__()
        
        assert kernelSize % 2 == 1, "Kernel size must be odd!"
        
        self.device = device
        
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        
        #self.tok_embedding = nn.Embedding(inputDim,embDim)
        self.inp2hid = nn.Linear(inputDim,embDim)                              #the first fc layer has dims out = emb dim
        self.posEmbedding = nn.Embedding(maxLenght,embDim)
        
        self.emb2hid = nn.Linear(embDim,hiddenDim)
        self.hid2emb = nn.Linear(hiddenDim,embDim)
        
        self.convs = nn.ModuleList([nn.Conv1d(in_channels = hiddenDim, 
                                              out_channels = 2*hiddenDim, 
                                              kernel_size = kernelSize, 
                                              padding = (kernelSize - 1) // 2)
                                    for _ in range(nLayers)])
        
        self.batchNormConvs = nn.ModuleList([nn.BatchNorm1d(2*hiddenDim,momentum=0.1,track_running_stats=False) for _ in range(nLayers)])
        
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self,curves):
        
        #curves = [batch size, curves len, curves components]
        
        batchSize = curves.shape[0]
        l = curves.shape[1]
        
        #create position tensor
        pos = torch.arange(0,l).unsqueeze(0).repeat(batchSize, 1).to(self.device)
        
        #pos = [0, 1, 2, 3, ..., curves len - 1]
        
        #pos = [batch size, curves len]
        
        #hidden points and embedded positions
        
        #tok_embedded = self.tok_embedding(src)
        
        x = curves.clone()
                
        #print(f"x shape {x.shape}")
        
        pointsHidden = self.inp2hid(x)
        posEmbedded = self.posEmbedding(pos)
        
        #tok_embedded = pos_embedded = [batch size, src len, emb dim]
        
        #combine embedding pos with hidden curves by elementwise summing
        embedded = self.dropout(pointsHidden + posEmbedded)
        #embedded = pointsHidden + posEmbedded
        
        #embedded = [batch size, src len, emb dim]
        
        #pass embedded through linear layer to convert from emb dim to hid dim
        convInput = self.emb2hid(embedded)
        
        #convInput = [batch size, src len, hid dim]
        
        #permute for convolutional layer (hidden dim became the number of input channels)
        convInput = convInput.permute(0,2,1) 
        
        #convInput = [batch size, hid dim, src len]
        
        #begin convolutional blocks
        
        for i,(conv,bn) in enumerate(zip(self.convs,self.batchNormConvs)):
        
            #pass through convolutional layer
            conved = conv(self.dropout(convInput))
            #conved = conv(convInput)

            #conved = [batch size, 2 * hid dim, src len]
            
            conved = bn(conved)

            #pass through GLU activation function
            conved = F.glu(conved, dim = 1)

            #conved = [batch size, hid dim, src len]
            
            #apply residual connection
            conved = (conved + convInput) * self.scale
            #conved = (conved + convInput)

            #conved = [batch size, hid dim, src len]
            
            #print(f"shape in encoder {conved.shape}")
            
            #set conv_input to conved for next loop iteration
            convInput = conved
        
        #end convolutional blocks
        
        #permute and convert back to emb dim
        conved = self.hid2emb(conved.permute(0, 2, 1))
        
        #conved = [batch size, src len, emb dim]
        
        #elementwise sum output (conved) and input (embedded) to be used for attention
        combined = (conved + embedded) * self.scale
        #combined = (conved + embedded)
        
        #combined = [batch size, src len, emb dim]
        
        return conved, combined
    

class DecoderCNN(nn.Module):
    def __init__(self,inputDim,outputDim,embDim,hiddenDim,nLayers,kernelSize,dropout,trgPadIdx,device,maxLength = 100,k=3):
        super().__init__()
        
        self.kernelSize = kernelSize
        self.trgPadIdx = trgPadIdx
        self.k = 3
        self.u = torch.cat((torch.tensor([0.0]*(self.k+1)),torch.tensor([1.0]*(self.k+1)))) 
        self.device = device
        
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        
        self.inp2hid = nn.Linear(inputDim,embDim)                              #the first fc layer has dims out = emb dim
        self.posEmbedding = nn.Embedding(maxLength,embDim)
        
        self.emb2hid = nn.Linear(embDim,hiddenDim)
        self.hid2emb = nn.Linear(hiddenDim,embDim)
        
        self.attnhid2emb = nn.Linear(hiddenDim,embDim)
        self.attnemb2hid = nn.Linear(embDim,hiddenDim)
        
        self.fct = nn.Linear(embDim, outputDim)                                #fc for parametrization t
        
        self.convs = nn.ModuleList([nn.Conv1d(in_channels = hiddenDim, 
                                              out_channels = 2*hiddenDim, 
                                              kernel_size = kernelSize)
                                    for _ in range(nLayers)])
        
        self.batchNormConvs = nn.ModuleList([nn.BatchNorm1d(2*hiddenDim,momentum=0.1,track_running_stats=False) for _ in range(nLayers)])
        
        self.dropout = nn.Dropout(dropout)
        
    def calculateAttention(self,embedded,conved,encoderConved,encoderCombined):
        
        #embedded = [batch size, trg len, emb dim]
        #conved = [batch size, hid dim, trg len]
        #encoder_conved = encoder_combined = [batch size, src len, emb dim]
        
        #permute and convert back to emb dim
        convedEmb = self.attnhid2emb(conved.permute(0, 2, 1))
        
        #conved_emb = [batch size, trg len, emb dim]
        
        combined = (convedEmb + embedded) * self.scale
        #combined = (convedEmb + embedded)
        
        #combined = [batch size, trg len, emb dim]
                
        energy = torch.matmul(combined, encoderConved.permute(0, 2, 1))
        
        #energy = [batch size, trg len, src len]
        
        attention = F.softmax(energy, dim=2)
        
        #attention = [batch size, trg len, src len]
            
        attendedEncoding = torch.matmul(attention, encoderCombined)
        
        #attended_encoding = [batch size, trg len, emd dim]
        
        #convert from emb dim -> hid dim
        attendedEncoding = self.attnemb2hid(attendedEncoding)
        
        #attended_encoding = [batch size, trg len, hid dim]
        
        #apply residual connection
        attendedCombined = (conved + attendedEncoding.permute(0, 2, 1)) * self.scale
        #attendedCombined = (conved + attendedEncoding.permute(0, 2, 1))
        
        #attended_combined = [batch size, hid dim, trg len]
        
        return attention, attendedCombined
    
    
    def forward(self,curves,encoderConved,encoderCombined):
        
        #curves = [batch size, curves len, curves components]
        #encoder_conved = encoder_combined = [batch size, src len, emb dim]
                
        batchSize = curves.shape[0]
        l = curves.shape[1]
        p = curves.shape[2]
            
        #create position tensor
        pos = torch.arange(0,l).unsqueeze(0).repeat(batchSize, 1).to(self.device)
        
        #pos = [batch size, curves len]
        
        #hidden points and embedded positions
        
        x = curves.clone()
        
        #tok_embedded = self.tok_embedding(src)
        
        pointsHidden = self.inp2hid(x)
        posEmbedded = self.posEmbedding(pos)
        
        #pointsHidden = [batch size, trg len, emb dim]
        #posEmbedded = [batch size, trg len, emb dim]
        
        #combine embeddings by elementwise summing
        embedded = self.dropout(pointsHidden + posEmbedded)
        #embedded = pointsHidden + posEmbedded
        
        #embedded = [batch size, trg len, emb dim]
        
        #pass embedded through linear layer to go through emb dim -> hid dim
        convInput = self.emb2hid(embedded)
        
        #convInput = [batch size, trg len, hid dim]
        
        #permute for convolutional layer
        convInput = convInput.permute(0,2,1) 
        
        #convInput = [batch size, hid dim, trg len]
        
        batchSize = convInput.shape[0]
        hiddenDim = convInput.shape[1]
        
        for i,(conv,bn) in enumerate(zip(self.convs,self.batchNormConvs)):
        
            #apply dropout
            convInput = self.dropout(convInput)
        
            #need to pad so decoder can't "cheat"
            padding = torch.zeros(batchSize, 
                                  hiddenDim, 
                                  self.kernelSize - 1).fill_(self.trgPadIdx).to(self.device)
                
            paddedConvInput = torch.cat((padding, convInput), dim = 2)
        
            #paddedConvInput = [batch size, hid dim, trg len + kernel size - 1]
        
            #pass through convolutional layer
            conved = conv(paddedConvInput)
            
            conved = bn(conved)

            #conved = [batch size, 2 * hid dim, trg len]
            
            #pass through GLU activation function
            conved = F.glu(conved, dim = 1)
            
            #print(f"prima dell'attenzione {conved.shape}")

            #conved = [batch size, hid dim, trg len]
            
            #calculate attention
            attention, conved = self.calculateAttention(embedded, 
                                                         conved, 
                                                         encoderConved, 
                                                         encoderCombined)
            
            #attention = [batch size, trg len, src len]
            
            #print(f"dopo l'attenzione {conved.shape}")
            
            #apply residual connection
            conved = (conved + convInput) * self.scale
            #conved = (conved + convInput)
            
            #conved = [batch size, hid dim, trg len]
            
            #set conv_input to conved for next loop iteration
            convInput = conved
            
        conved = self.hid2emb(conved.permute(0,2,1))
         
        #conved = [batch size, trg len, emb dim]
            
        delta = self.fct(self.dropout(conved))
        delta = delta.squeeze()
    
        delta = F.softplus(delta)
        #delta[:,0] = 0.0
        
        #print(f"delta shape {delta.shape}")
        
        #PP1. Accumulation & Rescaling Layer
        
        t = torch.cumsum(delta,1)
        #t = torch.cat((torch.zeros(t.shape[0],1).to(self.device),t),axis=1)
        tmin = torch.min(t,1)[0].view(-1,1)
        tmax = torch.max(t,1)[0].view(-1,1)
        t = torch.div(t-tmin,tmax-tmin)
        
        if len(t.unique()) < len(t):
            print("t si ripete")
        
        #PP2. Approximation Layer
        
        A = NDPWithBatch2(t,self.k,self.u,self.device)
        
        c = torch.zeros((batchSize,A.shape[2],p)).to(self.device)
        splines = torch.zeros((batchSize,l,p)).to(self.device)
        
        c = computeControlPointsWithBatch2(A[:,1:-1],curves[:,1:-1])
        
        splines = torch.matmul(A,c)
        
        return splines,t,curves,attention
            
        
"""Train SEQ2SEQ model"""

def trainSeq2Seq(seq2seq,optim,criterion,trainingset,device,epochs=10,batchSize=50):
    losses = list()
    avghd = list()
    
    nCurves = trainingset.shape[0]
    p = trainingset.shape[2]
    
    dataloader = DataLoader(trainingset,batch_size=batchSize,shuffle=True,pin_memory=True)
    
    start = timer()
    
    for e in range(epochs):
    
        """Training"""
    
        seq2seq.train()
        
        runningLoss = 0.0
        
        for i,data in enumerate(dataloader,0):
        
            input = data.to(device)
        
            optim.zero_grad()
            
            splines,t,curves = seq2seq(input)
            
            loss = criterion(curves,splines)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(seq2seq.parameters(),0.1)
            
            optim.step()
        
            runningLoss += loss.item()
            
            #del loss,splines,t,curves
        
            if i%(nCurves/(batchSize*2)) == (nCurves/(batchSize*2))-1:
                print(f"Epoca: {e+1}, Batch da {int(i+1-(nCurves/(batchSize*2)))} a {i}, Loss: {runningLoss/(nCurves/(batchSize*2))}")
                losses.append(runningLoss/(nCurves/(batchSize*2)))
                runningLoss = 0.0  
         
    end = timer()
    tElapsed = end-start
    print(f"Tempo trascorso: {tElapsed}s")
    
    return seq2seq

