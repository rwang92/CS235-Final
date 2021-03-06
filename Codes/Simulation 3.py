# import related packages

import torch
import numpy as np
from torch.autograd import Variable
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
%matplotlib inline
from torch.utils.data import DataLoader

# construct soft decision tree 

class SoftDecisionTree(nn.Module):

    def __init__ (self, in_dim, out_dim):
        super(SoftDecisionTree, self) .__init__()
        
        # define the nodes

        self.node = nn.Sigmoid()
        
        self.fc1 = nn.Linear(in_dim, 1)
        
        self.fc11 = nn.Linear(in_dim, 1)
        self.fc12 = nn.Linear(in_dim, 1)
        
        self.fc111 = nn.Linear(in_dim, 1)
        self.fc112 = nn.Linear(in_dim, 1)
        self.fc121 = nn.Linear(in_dim, 1)
        self.fc122 = nn.Linear(in_dim, 1)
        
        self.fc1111 = nn.Linear(in_dim, 1)
        self.fc1112 = nn.Linear(in_dim, 1)
        self.fc1121 = nn.Linear(in_dim, 1)
        self.fc1122 = nn.Linear(in_dim, 1)
        
        self.fc1211 = nn.Linear(in_dim, 1)
        self.fc1212 = nn.Linear(in_dim, 1)
        self.fc1221 = nn.Linear(in_dim, 1)
        self.fc1222 = nn.Linear(in_dim, 1)
        
        self.para1111 = nn.Linear(in_dim, 1)
        self.para1110 = nn.Linear(in_dim, 1)
        self.para1101 = nn.Linear(in_dim, 1)
        self.para1100 = nn.Linear(in_dim, 1)
        
        self.para1011 = nn.Linear(in_dim, 1)
        self.para1010 = nn.Linear(in_dim, 1)
        self.para1001 = nn.Linear(in_dim, 1)
        self.para1000 = nn.Linear(in_dim, 1)
        
        self.para0111 = nn.Linear(in_dim, 1)
        self.para0110 = nn.Linear(in_dim, 1)
        self.para0101 = nn.Linear(in_dim, 1)
        self.para0100 = nn.Linear(in_dim, 1)
        
        self.para0011 = nn.Linear(in_dim, 1)
        self.para0010 = nn.Linear(in_dim, 1)
        self.para0001 = nn.Linear(in_dim, 1)
        self.para0000 = nn.Linear(in_dim, 1)
        
        #add trainable parameters in each node to adjust the level of "soft"
        
        self.beta1 = nn.Parameter(torch.randn(1))
        self.beta2 = nn.Parameter(torch.randn(1))
        self.beta4 = nn.Parameter(torch.randn(1))
        self.beta8 = nn.Parameter(torch.randn(1))
        
        # define the forward propagating process
        
    def forward(self,x):
        p1=self.fc1(x)
        p1=self.node(p1*self.beta1)
                
        p11=self.fc11(x)
        p11=self.node(p11*self.beta2)
        p12=self.fc12(x)
        p12=self.node(p12*self.beta2)
      
        p111=self.fc111(x)
        p111=self.node(p111*self.beta4)
        p112=self.fc112(x)
        p112=self.node(p112*self.beta4)
        p121=self.fc121(x)
        p121=self.node(p121*self.beta4)
        p122=self.fc122(x)
        p122=self.node(p122*self.beta8)
        
        p1111=self.fc1111(x)
        p1111=self.node(p1111*self.beta8)
        p1112=self.fc1112(x)
        p1112=self.node(p1112*self.beta8)
        p1121=self.fc1121(x)
        p1121=self.node(p1121*self.beta8)
        p1122=self.fc1122(x)
        p1122=self.node(p1122*self.beta8)
        p1211=self.fc1211(x)
        p1211=self.node(p1211*self.beta8)
        p1212=self.fc1212(x)
        p1212=self.node(p1212*self.beta8)
        p1221=self.fc1221(x)
        p1221=self.node(p1221*self.beta8)
        p1222=self.fc1222(x)
        p1222=self.node(p1222*self.beta8)
        
        
        a1=p1
        a0=1-p1
        
        a11=p1*p11
        a10=p1*(1-p11)
        a01=(1-p1)*p12
        a00=(1-p1)*(1-p12)
        
        a111=p1*p11*p111
        a110=p1*p11*(1-p111)
        a101=p1*(1-p11)*p112
        a100=p1*(1-p11)*(1-p112)
        a011=(1-p1)*p12*p121
        a010=(1-p1)*p12*(1-p121)
        a001=(1-p1)*(1-p12)*p122
        a000=(1-p1)*(1-p12)*(1-p122)
        
        a1111=p1*p11*p111*p1111
        a1110=p1*p11*p111*(1-p1111)
        
        a1101=p1*p11*(1-p111)*p1112
        a1100=p1*p11*(1-p111)*(1-p1112)

        a1011=p1*(1-p11)*p112*p1121
        a1010=p1*(1-p11)*p112*(1-p1121)
        
        a1001=p1*(1-p11)*(1-p112)*p1121
        a1000=p1*(1-p11)*(1-p112)*(1-p1121)
        
        a0111=(1-p1)*p12*p121*p1211
        a0110=(1-p1)*p12*p121*(1-p1211)
        
        a0101=(1-p1)*p12*(1-p121)*p1212
        a0100=(1-p1)*p12*(1-p121)*(1-p1212)
        
        a0011=(1-p1)*(1-p12)*p122*p1221
        a0010=(1-p1)*(1-p12)*p122*(1-p1221)
        
        a0001=(1-p1)*(1-p12)*(1-p122)*p1222
        a0000=(1-p1)*(1-p12)*(1-p122)*(-p1222)
        
        mu1111=self.para1111(x)
        mu1110=self.para1110(x)
        mu1101=self.para1101(x)
        mu1100=self.para1100(x)
        mu1011=self.para1011(x)
        mu1010=self.para1010(x)
        mu1001=self.para1001(x)
        mu1000=self.para1000(x)
        mu0111=self.para0111(x)
        mu0110=self.para0110(x)
        mu0101=self.para0101(x)
        mu0100=self.para0100(x)
        mu0011=self.para0011(x)
        mu0010=self.para0010(x)
        mu0001=self.para0001(x)
        mu0000=self.para0000(x)
        
        return a1111*mu1111+a1110*mu1110+a1101*mu1101+a1100*mu1100+a1011*mu1011+a1010*mu1010+a1001*mu1001+a1000*mu1000+a0111*mu0111+a0110*mu0110+a0101*mu0101+a0100*mu0100+a0011*mu0011+a0010*mu0010+a0001*mu0001+a0000*mu0000
        
# construct the SNN

class simplenet(nn.Module):

# define the layers

    def __init__ (self, in_dim, hidden1_dim, hidden2_dim,hidden3_dim , out_dim):
        super(simplenet, self) .__init__()
        self.layer1=nn.Linear(in_dim, hidden1_dim)
        self.layer2=nn.Linear(hidden1_dim, hidden2_dim)
        self.layer3=nn.Linear(hidden2_dim, hidden3_dim)
        self.layer4=nn.Linear(hidden3_dim, out_dim)
        self.acti=nn.SELU(True)
        
# define the forward propagating process
        
    def forward(self,x):
        x=self.layer1(x)
        x=self.acti(x)
        x=self.layer2(x)
        x=self.acti(x)
        x=self.layer3(x)
        x=self.acti(x)
        x=self.layer4(x)
        return x
        
# define single hidden layer neural networks:

class simplenet2(nn.Module):
    def __init__ (self, in_dim, hidden1_dim, out_dim):
        super(simplenet2, self) .__init__()
        self.layer1=nn.Linear(in_dim, hidden1_dim)
        self.layer2=nn.Linear(hidden1_dim, out_dim)
        self.acti = nn.Sigmoid()
        
    def forward(self,x):
        x=self.layer1(x)
        x=self.acti(x)
        x=self.layer2(x)
        return x
 
#  generate simulation data 3


  #x_train= np.random.uniform(low=-30,high=30,size=(5000,2))
  
x0_train= np.random.normal(0,1,size=(1250,2))
x10_train= np.random.normal(10,1,size=(1250,2))

x20_train= np.random.normal(20,1,size=(1250,2))
x20_train[:,1]=x20_train[:,1]*(-1)
x30_train= np.random.normal(30,1,size=(1250,2))
x30_train[:,0]=x30_train[:,0]*(-1)

x_train=np.concatenate((x0_train,x10_train,x20_train,x30_train),axis=0)
  
y1_train=1+0.01*(7*x_train[:,0]+2*x_train[:,1]+4*x_train[:,0]*x_train[:,1]+10*np.exp(-2*((x_train[:,0]-10)**2+(x_train[:,1]-10)**2))+30*np.exp(-2*(x_train[:,0]**2+x_train[:,1]**2))+10*np.exp(-2*((x_train[:,0]-20)**2+(x_train[:,1]+20)**2))+10*np.exp(-2*((x_train[:,0]+30)**2+(x_train[:,1]-30)**2)))+np.random.normal(0,1,size=(5000,1)).T  
    
x0_test= np.random.normal(0,1,size=(250,2))
x10_test= np.random.normal(10,1,size=(250,2))

x20_test= np.random.normal(20,1,size=(250,2))
x20_test[:,1]=x20_test[:,1]*(-1)
x30_test= np.random.normal(30,1,size=(250,2))
x30_test[:,0]=x30_test[:,0]*(-1)

x_test=np.concatenate((x0_test,x10_test,x20_test,x30_test),axis=0)
  
y1_test=1+0.01*(7*x_test[:,0]+2*x_test[:,1]+4*x_test[:,0]*x_test[:,1]+10*np.exp(-2*((x_test[:,0]-10)**2+(x_test[:,1]-10)**2))+30*np.exp(-2*(x_test[:,0]**2+x_test[:,1]**2))+10*np.exp(-2*((x_test[:,0]-20)**2+(x_test[:,1]+20)**2))+10*np.exp(-2*((x_test[:,0]+30)**2+(x_test[:,1]-30)**2)))+np.random.normal(0,1,size=(1000,1)).T  

# load data in PyTorch
 
train1=np.concatenate((x_train, y1_train.T), axis=1)
train1=torch.from_numpy(train1)
train1=train1.type(torch.FloatTensor)
test1=np.concatenate((x_test, y1_test.T), axis=1)
test1=torch.from_numpy(test1)
test1=test1.type(torch.FloatTensor)

train_data1 = DataLoader(train1, batch_size=5000, shuffle=True)
test_data1 = DataLoader(test1, batch_size=1000, shuffle=False)
 
# import the HME
 
net=SoftDecisionTree(2,1)
criterion = nn.MSELoss()
 
# start training
 
losses = []
acces = []
eval_losses = []
eval_acces = []
optimzier = torch.optim.Adam(net.parameters(), lr = 1e-2) # use Adam as optimizer

for e in range(10000):
    train_loss = 0
    net.train()
    for d in train_data1:
      xs=d[:,0:2]        
      y = Variable(d[:,2])
      x = Variable(xs)
      xp = Variable(xs,requires_grad=True)
      out = net(x).view(-1)
      loss = criterion(out,y)
      optimzier.zero_grad()
      loss.backward()
      optimzier.step()
      train_loss += loss.data

    losses.append(train_loss / len(train_data1))
        
    # swicth to testing mode
     
    eval_loss = 0
    net.eval()
    for c in test_data1:
      xs=c[:,0:2]
      y = Variable(c[:,2])
      x = Variable(xs)
      xp=Variable(xs,requires_grad=True)
      out = net(x).view(-1)
      loss = criterion(out,y)
      eval_loss += loss.data
        
    eval_losses.append(eval_loss / len(test_data1))

    # print the MSE in each 200 epoch
    if e%200 == 0: 
      print('HME: epoch: {}, Train Loss: {:.6f}, Eval Loss: {:.6f}'.format(e, train_loss / len(train_data1),eval_loss / len(test_data1)))
        
 # import the SNN in CUDA mode
 
net=simplenet(2,30,30,20,1).cuda()
criterion = nn.MSELoss()
 
# start training
 
losses = []
acces = []
eval_losses = []
eval_acces = []
optimzier = torch.optim.Adam(net.parameters(), lr = 1e-2) # use Adam as optimizer

for e in range(10000):
    train_loss = 0
    net.train()
    for d in train_data1:
      xs=d[:,0:2]        
      y = Variable(d[:,2]).cuda()
      x = Variable(xs).cuda()
      out = net(x).view(-1)
      loss = criterion(out,y)
      optimzier.zero_grad()
      loss.backward()
      optimzier.step()
      train_loss += loss.data

    losses.append(train_loss / len(train_data1))
        
    # swicth to testing mode
     
    eval_loss = 0
    net.eval()
    for c in test_data1:
      xs=c[:,0:2]
      y = Variable(c[:,2]).cuda()
      x = Variable(xs).cuda()
      out = net(x).view(-1)
      loss = criterion(out,y)
      eval_loss += loss.data
        
    eval_losses.append(eval_loss / len(test_data1))

    # print the MSE in each 200 epoch
    if e%200 == 0: 
      print('SNN: epoch: {}, Train Loss: {:.6f}, Eval Loss: {:.6f}'.format(e, train_loss / len(train_data1),eval_loss / len(test_data1)))
 
 # import the one hidden layer NN in CUDA mode
 
net=simplenet2(2,100,1).cuda()
criterion = nn.MSELoss()
 
# start training
 
losses = []
acces = []
eval_losses = []
eval_acces = []
optimzier = torch.optim.Adam(net.parameters(), lr = 1e-2) # use Adam as optimizer

for e in range(10000):
    train_loss = 0
    net.train()
    for d in train_data1:
      xs=d[:,0:2]        
      y = Variable(d[:,2]).cuda()
      x = Variable(xs).cuda()
      out = net(x).view(-1)
      loss = criterion(out,y)
      optimzier.zero_grad()
      loss.backward()
      optimzier.step()
      train_loss += loss.data

    losses.append(train_loss / len(train_data1))
        
    # swicth to testing mode
     
    eval_loss = 0
    net.eval()
    for c in test_data1:
      xs=c[:,0:2]
      y = Variable(c[:,2]).cuda()
      x = Variable(xs).cuda()
      out = net(x).view(-1)
      loss = criterion(out,y)
      eval_loss += loss.data
        
    eval_losses.append(eval_loss / len(test_data1))

    # print the MSE in each 200 epoch
    if e%200 == 0: 
      print('NN: epoch: {}, Train Loss: {:.6f}, Eval Loss: {:.6f}'.format(e, train_loss / len(train_data1),eval_loss / len(test_data1)))
