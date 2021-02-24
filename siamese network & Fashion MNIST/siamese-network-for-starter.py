# %% [code]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %% [code]
data_train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
data_train.head()

# %% [code]
X_full = data_train.iloc[:,1:]
y_full = data_train.iloc[:,:1]
x_train, x_test, y_train, y_test = train_test_split(X_full, y_full, test_size = 0.05)
x_train = x_train.values.reshape(-1, 1,28, 28).astype('float32') / 255.
x_test = x_test.values.reshape(-1, 1,28, 28).astype('float32') / 255.
y_train.label.unique()
np.bincount(y_train.label.values),np.bincount(y_test.label.values)

# %% [code]
class mydataset(Dataset):
    def __init__(self,x_data,y_data):
        self.x_data = x_data
        self.y_data = y_data.label.values
    def __len__(self):
        return len(self.x_data)
    def __getitem__(self,idx):
        img1 = self.x_data[idx]
        y1 = self.y_data[idx]
        if np.random.rand() < 0.5:  
            idx2 = np.random.choice(np.arange(len(self.y_data))[self.y_data==y1],1)
        else:
            idx2 = np.random.choice(np.arange(len(self.y_data))[self.y_data!=y1],1)
        img2 = self.x_data[idx2[0]]
        y2 = self.y_data[idx2[0]]
        label = [0] if y1==y2 else [1]
        return torch.FloatTensor(img1),torch.FloatTensor(img2),torch.FloatTensor(label),y1
train_dataset = mydataset(x_train,y_train)
train_dataloader = DataLoader(dataset = train_dataset,batch_size=8)
val_dataset = mydataset(x_test,y_test)
val_dataloader = DataLoader(dataset = val_dataset,batch_size=8)
for idx,(img1,img2,target,_) in enumerate(train_dataloader):
    fig, axs = plt.subplots(2, img1.shape[0], figsize = (12, 6))
    for idx,(ax1,ax2) in enumerate(axs.T):
        ax1.imshow(img1[idx,0,:,:].numpy(),cmap='gray')
        ax1.set_title('image A')
        ax2.imshow(img2[idx,0,:,:].numpy(),cmap='gray')
        ax2.set_title('{}'.format('same' if target[idx,0]==0 else 'different'))
    break


# %% [code]
class siamese(nn.Module):
    def __init__(self,z_dimensions=2):
        super(siamese,self).__init__()
        self.feature_net = nn.Sequential(
            nn.Conv2d(1,8,kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Conv2d(8,8,kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),
            nn.Conv2d(8,16,kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,16,kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Conv2d(16,1,kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True)
        )
        self.linear1 = nn.Linear(49,49)
        self.linear2 = nn.Linear(49,z_dimensions)
    def forward(self,x):
        x = self.feature_net(x)
        x = x.view(x.shape[0],-1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
model = siamese(z_dimensions=8).to(device)
# model.load_state_dict(torch.load('../working/saimese.pth'))
optimizor = torch.optim.Adam(model.parameters(),lr=0.001)
def contrastive_loss(pred1,pred2,target):
    MARGIN = 2
    euclidean_dis = F.pairwise_distance(pred1,pred2)
    target = target.view(-1)
    loss = (1-target)*torch.pow(euclidean_dis,2) + target * torch.pow(torch.clamp(MARGIN-euclidean_dis,min=0),2)
    loss = torch.mean(loss)
    return loss


# %% [code]
train_dataset = mydataset(x_train,y_train)
train_dataloader = DataLoader(dataset = train_dataset,batch_size=128)
val_dataset = mydataset(x_test,y_test)
val_dataloader = DataLoader(dataset = val_dataset,batch_size=128)

# %% [code]
for e in range(50):
    history = []
    for idx,(img1,img2,target,_) in enumerate(train_dataloader):
        img1 = img1.to(device)
        img2 = img2.to(device)
        target = target.to(device)
        
        pred1 = model(img1)
        pred2 = model(img2)
        loss = contrastive_loss(pred1,pred2,target)

        optimizor.zero_grad()
        loss.backward()
        optimizor.step()
        
        loss = loss.detach().cpu().numpy()
        history.append(loss)
        train_loss = np.mean(history)
    history = []
    with torch.no_grad():
        for idx,(img1,img2,target,_) in enumerate(val_dataloader):
            img1 = img1.to(device)
            img2 = img2.to(device)
            target = target.to(device)

            pred1 = model(img1)
            pred2 = model(img2)
            loss = contrastive_loss(pred1,pred2,target)

            loss = loss.detach().cpu().numpy()
            history.append(loss)
            val_loss = np.mean(history)
    print(f'train_loss:{train_loss},val_loss:{val_loss}')

# %% [code]
torch.save(model.state_dict(),'saimese.pth')

# %% [code]
x = [];y = [];
with torch.no_grad():
    for idx,(img1,img2,target,y1) in enumerate(val_dataloader):
        img1 = img1.to(device)
        img2 = img2.to(device)
        target = target.to(device)

        pred1 = model(img1)
        pred2 = model(img2)
        loss = contrastive_loss(pred1,pred2,target)

        x.append(pred1.detach().cpu().numpy())
        y.append(y1.detach().cpu().numpy())


# %% [code]
X = np.concatenate(x,axis=0)
y = np.concatenate(y,axis=0)
y = y.reshape(-1)

# %% [code]
from sklearn import manifold
'''X是特征，不包含target;X_tsne是已经降维之后的特征'''
tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
X_tsne = tsne.fit_transform(X)
print("Org data dimension is {}. \
      Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
      
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
plt.figure(figsize=(8, 8))
for i in range(10):
    plt.scatter(X_norm[y==i][:,0],X_norm[y==i][:,1],alpha=0.3,label=f'{i}')
plt.legend()
