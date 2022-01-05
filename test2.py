import torch
from utils import *
import pdb

checkpoint = torch.load(r"D:\GaitRecognition\result\gaitpart_2942\checkpoints\iter-06000.pt")

all_result = checkpoint["all_result"]
pdb.set_trace()
data_visualization("result", all_result)
