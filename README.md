# Deep Understand about ONNX

In this project, I will use `torch` and `onnx` to provide from basic to complex understanding about ONNX

## How ONNX store params

Create a simple onnx model with one `Linear` layer using `torch`

``` python
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(128, 128)

    def forward(self, x):
        return self.layer(x)
```

Convert model to onnx and visualize in [netron](https://netron.app)

![basic_model](./assets/basic.png)

Get information about weight and bias with `read_onnx.py`

``` bash
layer.weight: (128, 128)
layer.bias: (128,)
```

And if you print in `numpy` format, you will get value of weight and bias