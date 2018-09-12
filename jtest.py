import matplotlib.pyplot as plt
from models import DispNetS, PoseExpNet
import torch
from torch.autograd import Variable
import time
import numpy as np
import os
import cv2
import glob
device_name= "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)


class time_lapse:
	def __init__(self):
		self.start=time.time()
		self.count=0
	def now(self):
		self.count += 1
		print("[%d]time elasped = %f"% (self.count,time.time()-self.start))

tl=time_lapse()
tl.now()


def init_disp_net(pretrained):
	weights = torch.load(pretrained, map_location=device_name)
	disp_net = DispNetS().to(device)
	disp_net.load_state_dict(weights['state_dict'])
	disp_net.eval()
	return disp_net

def dis_predict(disp_net, img):
	#an_input0 = cv2.imread(img, cv2.IMREAD_COLOR)
	an_input0 = cv2.imread(img, cv2.IMREAD_UNCHANGED)
	# convert RGB <-> BGR
	an_input0 = an_input0[:,:,::-1]
	
	h, w, c = an_input0.shape
	print(h, w, c)
	ww=300
	hh=int(h*ww/w+0.5)
	an_input0=cv2.resize(an_input0, (ww,hh))
	#an_input0 = transform.resize(an_input0, (hh, ww))
	an_input = np.transpose(an_input0, (2,0,1))
	an_input = torch.from_numpy(an_input)
	an_input = an_input.unsqueeze(0)

	#print(an_input)
	an_input = ( (an_input.float()/255-0.5)/0.5)
	an_input = an_input.to(device)
	tl.now()
	pred_disp = disp_net(an_input).cpu().detach().numpy()[0,0]
	pred_disp = (pred_disp * 2 +0.5)*255*100
	return an_input0, pred_disp/pred_disp.max()

def main():
	pretrained = "pretrained/dispnet_model_best.pth.tar"
	inputs_dir = "samples"
	disp_net = init_disp_net(pretrained)
	imgs=[ f for f in glob.glob(os.path.join(inputs_dir,"*")) if os.path.isfile(f) ]
	print(imgs)
	an_inputs, outputs=[], []
	for img in imgs:
		an_input0, pred_disp=dis_predict(disp_net, img)
		an_inputs.append(an_input0)
		outputs.append(pred_disp)
		
	tl.now()
	nrows = len(an_inputs)
	fig, axes = plt.subplots(nrows,2)
	for i in range(nrows):
		axes[i][1].imshow(an_inputs[i])
		axes[i][0].imshow(outputs[i])
	plt.show()


if __name__=="__main__":
	main()
