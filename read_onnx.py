import onnx

model = onnx.load("sample_model.onnx")
graph_init = model.graph.initializer
for block in graph_init:
    param = onnx.numpy_helper.to_array(block)
    print(f"{block.name}: {param.shape}")
