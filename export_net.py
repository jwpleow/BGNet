"""
This code is used to convert the pytorch model into an onnx format model.
"""

from models.bgnet_plus import BGNet_Plus
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from datasets.data_io import get_transform
import torch.utils.data
import torch.onnx


model = BGNet_Plus().cuda()

checkpoint = torch.load('models/Sceneflow-IRS-BGNet-Plus.pth',
                        map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint)
model.eval()


left = torch.randint(0, 256, (1, 1, 384, 640)).float().cuda() # must be multiple of 64
right = torch.randint(0, 256, (1, 1, 384, 640)).float().cuda()
inputs = (left, right)


# traced_gpu = torch.jit.trace(model, inputs, strict=False)
traced_gpu = torch.jit.trace(model, inputs)
torch.jit.save(traced_gpu, "gpu_model.pt")
# torch.onnx.export(aanet, (left, right), "model.onnx", verbose=False, input_names=['left', 'right'], output_names=['disparity_pyramid'], opset_version=11)
