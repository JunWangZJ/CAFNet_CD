from __future__ import print_function
import torch
import numpy as np
from skimage import io, measure
from torch import nn
from preclassify import dicomp, hcluster,srad
import os
import matplotlib.pyplot as plt
import random
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from CSFNet_model import CSFNet
patch_size = 9
epochs =25
lr = 0.0001
gamma = 0.7
seed = 42

def image_normalize(data):
  import math
  _mean = np.mean(data)
  _std = np.std(data)
  npixel = np.size(data) * 1.0
  min_stddev = 1.0 / math.sqrt(npixel)
  return (data - _mean) / max(_std, min_stddev)


def image_padding(data,r):
  if len(data.shape)==3:
    data_new=np.lib.pad(data,((r,r),(r,r),(0,0)),'constant',constant_values=0)
    return data_new
  if len(data.shape)==2:
    data_new=np.lib.pad(data,r,'constant',constant_values=0)
    return data_new

def arr(length):
  arr=np.arange(length-1)
  #print(arr)
  random.shuffle(arr)
  #print(arr)
  return arr

def seed_everything(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True

seed_everything(seed)
device = 'cuda'
def createTrainingCubes(X, y, patch_size):
  # 给 X 做 padding
  margin = int((patch_size - 1) / 2)
  zeroPaddedX = image_padding(X, margin)
  # 把类别 uncertainty 的像素忽略
  ele_num1 = np.sum(y==1)
  ele_num2 = np.sum(y==2)
  patchesData_1 = np.zeros( (ele_num1, patch_size, patch_size, X.shape[2]) )
  patchesLabels_1 = np.zeros(ele_num1)

  patchesData_2 = np.zeros((ele_num2, patch_size, patch_size, X.shape[2]))
  patchesLabels_2 = np.zeros(ele_num2)

  patchIndex_1 = 0
  patchIndex_2 = 0
  for r in range(margin, zeroPaddedX.shape[0] - margin):
    for c in range(margin, zeroPaddedX.shape[1] - margin):
      # remove uncertainty pixels
      if y[r-margin, c-margin] == 1 :
        patch_1 = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
        patchesData_1[patchIndex_1, :, :, :] = patch_1
        patchesLabels_1[patchIndex_1] = y[r-margin, c-margin]
        patchIndex_1 = patchIndex_1 + 1
      elif y[r-margin, c-margin] == 2 :
        patch_2 = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
        patchesData_2[patchIndex_2, :, :, :] = patch_2
        patchesLabels_2[patchIndex_2] = y[r-margin, c-margin]
        patchIndex_2 = patchIndex_2 + 1
  patchesLabels_1 = patchesLabels_1-1
  patchesLabels_2 = patchesLabels_2-1

  #调用arr函数打乱数组
  arr_1=arr(len(patchesData_1))
  arr_2=arr(len(patchesData_2))


  train_len = 10000  # 设置训练集样本数
  pdata = np.zeros((train_len, patch_size, patch_size, X.shape[2]))
  plabels = np.zeros(train_len)

  for i in range(7000):
    pdata[i, :, :, :] = patchesData_1[arr_1[i], :, :, :]
    plabels[i] = patchesLabels_1[arr_1[i]]
  for j in range(7000, train_len):
    pdata[j, :, :, :] = patchesData_2[arr_2[j - 7000], :, :, :]
    plabels[j] = patchesLabels_2[arr_2[j - 7000]]

  return pdata, plabels



def createTestingCubes(X, patch_size):
  # 给 X 做 padding
  margin = int((patch_size - 1) / 2)
  zeroPaddedX = image_padding(X, margin)
  patchesData = np.zeros( (X.shape[0]*X.shape[1], patch_size, patch_size, X.shape[2]) )
  patchIndex = 0
  for r in range(margin, zeroPaddedX.shape[0] - margin):
    for c in range(margin, zeroPaddedX.shape[1] - margin):
      patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
      patchesData[patchIndex, :, :, :] = patch
      patchIndex = patchIndex + 1
  return patchesData



im1_path  = 'data/Yellow_River/Yellow_River_1.bmp'
im2_path  = 'data/Yellow_River/Yellow_River_2.bmp'
imgt_path = 'data/Yellow_River/Yellow_River_gt.bmp'
im1 = io.imread(im1_path).astype(np.float32)
im2 = io.imread(im2_path).astype(np.float32)
im_gt = io.imread(imgt_path).astype(np.float32)



# im1_path  = 'data/Sulzberger/Sulzberger2_1.bmp'
# im2_path  = 'data/Sulzberger/Sulzberger2_2.bmp'
# imgt_path = 'data/Sulzberger/Sulzberger2_gt.bmp'
# im1 = io.imread(im1_path)[:, :, 0].astype(np.float32)
# im2 = io.imread(im2_path)[:, :, 0].astype(np.float32)
# im_gt = io.imread(imgt_path)[:, :, 0].astype(np.float32)



im_di = dicomp(im1, im2) #得到去噪的差异图
ylen, xlen = im_di.shape
pix_vec = im_di.reshape([ylen*xlen, 1]) #得到去噪后的一维度的列向量

preclassify_lab = hcluster(pix_vec, im_di)
print('... ... hiearchical clustering finished !!!')


mdata = np.zeros([im1.shape[0], im1.shape[1], 3], dtype=np.float32)
mdata[:,:,0] = im1
mdata[:,:,1] = im2
mdata[:,:,2] = im_di
mlabel = preclassify_lab

x_train, y_train = createTrainingCubes(mdata, mlabel, patch_size)
x_train = x_train.transpose(0, 3, 1, 2)
print('... x train shape: ', x_train.shape)
print('... y train shape: ', y_train.shape)


x_test = createTestingCubes(mdata, patch_size)
x_test = x_test.transpose(0, 3, 1, 2)
print('... x test shape: ', x_test.shape)

""" Training dataset"""

class TrainDS(torch.utils.data.Dataset):
  def __init__(self):
    self.len = x_train.shape[0]
    self.x_data = torch.FloatTensor(x_train)
    self.y_data = torch.LongTensor(y_train)
  def __getitem__(self, index):

    return self.x_data[index], self.y_data[index]
  def __len__(self):
    # 返回文件数据的数目
    return self.len

# 创建 trainloader 和 testloader
trainset = TrainDS()
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, shuffle=True, num_workers=0)



criterion = nn.CrossEntropyLoss()
# optimizer
model = CSFNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)


def evaluate(gtImg, tstImg):
  gtImg[np.where(gtImg>128)] = 255
  gtImg[np.where(gtImg<128)] = 0
  tstImg[np.where(tstImg>128)] = 255
  tstImg[np.where(tstImg<128)] = 0
  [ylen, xlen] = gtImg.shape
  FA = 0.0
  MA = 0.0

  label_0 = np.sum(gtImg==0)*1.0
  label_1 = np.sum(gtImg==255)*1.0
  print(label_0)
  print(label_1)

  for j in range(0,ylen):
    for i in range(0,xlen):
      if gtImg[j,i]==0 and tstImg[j,i]!=0 :
        FA = FA+1
      if gtImg[j,i]!=0 and tstImg[j,i]==0 :
        MA = MA+1

  OE = FA+MA
  PCC = 1-OE/(ylen*xlen*1.0)
  PRE=((label_1+FA-MA)*label_1+(label_0+MA-FA)*label_0)/((ylen*xlen)*(ylen*xlen*1.0))
  KC=(PCC-PRE)/(1-PRE)
  print(' Change detection results ==>')
  print(' ... ... FP:  ', FA)
  print(' ... ... FN:  ', MA)
  print(' ... ... OE:  ', OE)
  print(' ... ... PCC: ', format(PCC*100, '.2f'))
  print(' ... ... KC: ', format(KC*100, '.2f'))


def postprocess1(res):
  res_new = res
  res = measure.label(res, connectivity=2)
  #print(res)
  num = res.max()
  #print(num)
  for i in range(1, num+1):
    idy, idx = np.where(res==i)
    if len(idy) <= 20:
      res_new[idy, idx] = 0
  return res_new
def postprocess(res):
  res_new = res
  res = measure.label(res, connectivity=2)
  #print(res)
  num = res.max()
  #print(num)
  for i in range(1, num+1):
    idy, idx = np.where(res==i)
    if len(idy) <= 20:
      res_new[idy, idx] = 0.5
  return res_new


outputs = np.zeros((ylen, xlen))
for epoch in range(epochs):
  epoch_loss = 0
  epoch_accuracy = 0
  for data, label in tqdm(train_loader):
    data = data.to(device)
    label = label.to(device)

    in_x, hamout, camout, catout, fuseout, output = model(data)
    loss = criterion(output, label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc = (output.argmax(dim=1) == label).float().mean()
    epoch_accuracy += acc / len(train_loader)
    epoch_loss += loss / len(train_loader)
  print(
      f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}\n"
  )

istrain=False
model.eval()
outputs = np.zeros((ylen, xlen))
for i in range(ylen):
  for j in range(xlen):
    if preclassify_lab[i, j] != 1.5 :
      outputs[i, j] = preclassify_lab[i, j]
    else:
      img_patch = x_test[i*xlen+j, :, :, :]
      img_patch = img_patch.reshape(1, img_patch.shape[0], img_patch.shape[1], img_patch.shape[2])
      img_patch = torch.FloatTensor(img_patch).to(device)
      in_x, hamout, camout, catout, fuseout, prediction = model(img_patch)
      prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
      outputs[i, j] = prediction+1
  if (i+1) % 50 == 0:
    print('... ... row', i+1, ' handling ... ...')

outputs = outputs-1

res = outputs * 255
res = postprocess(res)
evaluate(im_gt, res)

plt.imshow(res, 'gray')
plt.show()


