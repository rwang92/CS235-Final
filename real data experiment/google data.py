# import packages

import pandas as pd
import numpy as np
import time
from datetime import datetime
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.autograd import Variable
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
#from torchvision import datasets, transforms
%matplotlib inline
from torch.utils.data import DataLoader
import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

# Define the function for loading data

def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df


# Data loading (just load testing data)

train_df = load_df("../input/test_v2.csv")

# check the attributes

train_df.columns

# Extract target variable transaction Revenue and fill NA with 0

target = train_df['totals.transactionRevenue'].fillna(0).astype(float)
del train_df['totals.transactionRevenue']

# Visualize target variable

target_0 = target
sns.set(color_codes=True)
target_0=target_0.as_matrix()

sns.distplot(target_0) # with NAs
sns.distplot(target_0[target_0!=0]) # without NAs

# Take log and visualize log(target)

target = target.apply(lambda x: np.log1p(x))
target_1 = target.as_matrix()

sns.distplot(target_1) # with NAs
sns.distplot(target_1[target_1!=0]) # without NAs

# remove all the variables containing one outcome

columns = [col for col in train_df.columns if train_df[col].nunique() > 1]
train_df = train_df[columns]


merged_df = train_df
merged_df['diff_visitId_time'] = merged_df['visitId'] - merged_df['visitStartTime']
merged_df['diff_visitId_time'] = (merged_df['diff_visitId_time'] != 0).astype(int)
del merged_df['visitId']

format_str = '%Y%m%d' 
merged_df['formated_date'] = merged_df['date'].apply(lambda x: datetime.strptime(str(x), format_str))
merged_df['WoY'] = merged_df['formated_date'].apply(lambda x: x.isocalendar()[1])
merged_df['month'] = merged_df['formated_date'].apply(lambda x:x.month)
merged_df['quarter_month'] = merged_df['formated_date'].apply(lambda x:x.day//8)
merged_df['weekday'] = merged_df['formated_date'].apply(lambda x:x.weekday())

del merged_df['date']
del merged_df['formated_date']

merged_df['formated_visitStartTime'] = merged_df['visitStartTime'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
merged_df['formated_visitStartTime'] = pd.to_datetime(merged_df['formated_visitStartTime'])
merged_df['visit_hour'] = merged_df['formated_visitStartTime'].apply(lambda x: x.hour)

del merged_df['visitStartTime']
del merged_df['formated_visitStartTime']

for col in merged_df.columns:
    if col in ['fullVisitorId', 'month', 'quarter_month', 'weekday', 'visit_hour', 'WoY']: continue
    if merged_df[col].dtypes == object or merged_df[col].dtypes == bool:
        merged_df[col], indexer = pd.factorize(merged_df[col])

del merged_df['fullVisitorId']

train_df = merged_df

# Check the variable names again

train_df.columns

# Convert dataframe to matrix for training

train_set=merged_df.as_matrix()

# Construct HMC

class SoftDecisionTree(nn.Module):
    def __init__ (self, in_dim, out_dim):
        super(SoftDecisionTree, self) .__init__()
        
        self.node = nn.Sigmoid()
        #self.node = torch.sin()
        
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
                
        self.beta1 = nn.Parameter(torch.randn(1))
        self.beta2 = nn.Parameter(torch.randn(1))
        self.beta4 = nn.Parameter(torch.randn(1))
        self.beta8 = nn.Parameter(torch.randn(1))
        
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

# Construct SNN and NN

class simplenet(nn.Module):
    def __init__ (self, in_dim, hidden1_dim, hidden2_dim,hidden3_dim , out_dim):
        super(simplenet, self) .__init__()
        self.layer1=nn.Linear(in_dim, hidden1_dim)
        self.layer2=nn.Linear(hidden1_dim, hidden2_dim)
        self.layer3=nn.Linear(hidden2_dim, hidden3_dim)
        self.layer4=nn.Linear(hidden3_dim, out_dim)
        self.acti=nn.ReLU(True)
        
    def forward(self,x):
        x=self.layer1(x)
        x=self.acti(x)
        x=self.layer2(x)
        x=self.acti(x)
        x=self.layer3(x)
        x=self.acti(x)
        x=self.layer4(x)
        return x

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
 
# load the training data
 
target_y=target.as_matrix()

x_train= train_set[0:30000,:]
y1_train = target_y[0:30000].view(type=np.matrix)
y1_train = y1_train.T
x_test= train_set[30000:40000,:]
y1_test = target_y[30000:40000].view(type=np.matrix)
y1_test = y1_test.T
  
train1=np.concatenate((x_train, y1_train), axis=1)
train1=torch.from_numpy(train1)
train1=train1.type(torch.FloatTensor)

test1=np.concatenate((x_test, y1_test), axis=1)
test1=torch.from_numpy(test1)
test1=test1.type(torch.FloatTensor)

train_data1 = DataLoader(train1, batch_size=30000, shuffle=True) # take 30000 data points for training
test_data1 = DataLoader(test1, batch_size=10000, shuffle=False) # take 10000 data points for testing
  
# training HME

net=SoftDecisionTree(37,1).cuda()
criterion = nn.MSELoss()
  
losses = []
acces = []
eval_losses = []
eval_acces = []

optimzier = torch.optim.Adam(net.parameters(), lr = 1e-2)

for e in range(10000):
    train_loss = 0
    net.train()
    for d in train_data1:
        xs=d[:,0:37]
        y = Variable(d[:,37]).cuda()
        x = Variable(xs).cuda()
        out = net(x).view(-1)
        loss = criterion(out,y)
        optimzier.zero_grad()
        loss.backward()
        optimzier.step()
        train_loss += loss.data

        losses.append(train_loss / len(train_data1))
        
    eval_loss = 0
    net.eval()
    for c in test_data1:
        xs=c[:,0:37]
        y = Variable(c[:,37]).cuda()
        x = Variable(xs).cuda()
        out = net(x).view(-1)
        loss = criterion(out,y)
        eval_loss += loss.data
        
        eval_losses.append(eval_loss / len(test_data1))
    if e%200 == 0:
        print('HME:epoch: {}, Train Loss: {:.6f}, Eval Loss: {:.6f}'.format(e, train_loss / len(train_data1),eval_loss / len(test_data1)))

# training SNN

net=simplenet(37,30,30,20,1).cuda()
criterion = nn.MSELoss()
  
losses = []
acces = []
eval_losses = []
eval_acces = []

optimzier = torch.optim.Adam(net.parameters(), lr = 1e-2)

for e in range(10000):
    train_loss = 0
    net.train()
    for d in train_data1:
        xs=d[:,0:37]
        y = Variable(d[:,37]).cuda()
        x = Variable(xs).cuda()
        out = net(x).view(-1)
        loss = criterion(out,y)
        optimzier.zero_grad()
        loss.backward()
        optimzier.step()
        train_loss += loss.data

        losses.append(train_loss / len(train_data1))
        
    eval_loss = 0
    net.eval()
    for c in test_data1:
        xs=c[:,0:37]
        y = Variable(c[:,37]).cuda()
        x = Variable(xs).cuda()
        out = net(x).view(-1)
        loss = criterion(out,y)
        eval_loss += loss.data
        
        eval_losses.append(eval_loss / len(test_data1))
    if e%200 == 0:
        print('SNN:epoch: {}, Train Loss: {:.6f}, Eval Loss: {:.6f}'.format(e, train_loss / len(train_data1),eval_loss / len(test_data1)))

# training NN


net=simplenet2(37,100,1).cuda()
criterion = nn.MSELoss()

losses = []
acces = []
eval_losses = []
eval_acces = []

optimzier = torch.optim.Adam(net.parameters(), lr = 1e-2)
for e in range(10000):
    train_loss = 0
    net.train()  
    for d in train_data1:
        xs=d[:,0:37]
        y = Variable(d[:,37]).cuda()
        x = Variable(xs).cuda()
        out = net(x).view(-1)
        loss = criterion(out,y)
        optimzier.zero_grad()
        loss.backward()
        optimzier.step()

        train_loss += loss.data

    losses.append(train_loss / len(train_data1))
        
    eval_loss = 0
    net.eval()
    for c in test_data1:
        xs=c[:,0:37]
        y = Variable(c[:,37]).cuda()
        x = Variable(xs).cuda()
        out = net(x).view(-1)
        loss = criterion(out,y)
        eval_loss += loss.data
        
    eval_losses.append(eval_loss / len(test_data1))
      

    if e%200 == 0:
        print('NN:epoch: {}, Train Loss: {:.6f}, Eval Loss: {:.6f}'.format(e, train_loss / len(train_data1),eval_loss / len(test_data1)))
       


