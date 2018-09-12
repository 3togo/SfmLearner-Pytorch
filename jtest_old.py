from models import DispNetS, PoseExpNet
import torch
from torch.autograd import Variable
from skimage import io, transform
import time
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import cv2

class time_lapse:
	def __init__(self):
		self.start=time.time()
		self.count=0
	def now(self):
		self.count += 1
		print("[%d]time elasped = %f"% (self.count,time.time()-self.start))
tl=time_lapse()
tl.now()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
disp_net = DispNetS().to(device)
weights = torch.load("pretrained/dispnet_model_best.pth.tar", map_location='cpu')
disp_net.load_state_dict(weights['state_dict'])
disp_net.eval()

tgt_img0 = io.imread("samples/street1.jpeg")
h, w, c = tgt_img0.shape
#print(h, w, c)
print(h, w)
ww=600
hh=int(h*ww/w+0.5)
tgt_img0=cv2.resize(tgt_img0, (ww,hh))
print(ww,hh)
#tgt_img0 = transform.resize(tgt_img0, (hh, ww))
tgt_img0 = torch.from_numpy(tgt_img0)
tgt_img = np.transpose(tgt_img0, (2,0,1))
print(tgt_img.shape)
tgt_img = tgt_img.unsqueeze(0)

tgt_img = ( (tgt_img.float()/255-0.5)/0.5)
tgt_img = tgt_img.to(device)
tl.now()
pred_disp = disp_net(tgt_img).cpu().detach().numpy()[0,0]
pred_disp = (pred_disp * 2 +0.5)*255*100
#print("shape=", pred_disp.shape)
#print(pred_disp)
tl.now()
#plt.imshow(pred_disp/pred_disp.max())
#plt.show()
#nrows = len(inputs)
nrows=1
fig, axes = plt.subplots(nrows,2)
axes[0].imshow(tgt_img0)
axes[1].imshow(pred_disp/pred_disp.max())
plt.show()
