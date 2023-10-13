import torch
from torch import nn

torch.manual_seed(42)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(128, 128)

    def forward(self, x):
        return self.layer(x)
    
model = SimpleModel()
model.eval()
inputs = torch.randn(size=(1, 128))
outputs = model(inputs)
print(outputs)

torch.onnx.export(model,               # model being run
                  inputs,                         # model input (or a tuple for multiple inputs)
                  "sample_model.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=13,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})