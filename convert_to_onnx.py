import onnx, onnxruntime, torch
import numpy as np
from collections import OrderedDict
from train import modelAvg, modelTime, NUMBER_OF_MOVES

net = modelAvg().to('cpu')

state_dict = torch.load('model_avg.pth', map_location='cpu')
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    k = k.replace('module.', '')
    new_state_dict[k] = v

net.load_state_dict(new_state_dict)
net.eval()

move_input = torch.from_numpy(np.random.randint(0, 1+NUMBER_OF_MOVES, size=(1, 150), dtype=np.int64)).to(torch.int64)
mask_input = torch.from_numpy(np.random.randint(0, 2, size=(1, 150), dtype=np.bool_)).to(torch.bool)
onnx_program = torch.onnx.dynamo_export(net, move_input, mask_input)
onnx_program.save("model_avg.onnx")

onnx_model = onnx.load("model_avg.onnx")
onnx.checker.check_model(onnx_model)
print('Successfully saved base model')

onnx_input = onnx_program.adapt_torch_inputs_to_onnx(move_input, mask_input)
print(f"Input length: {len(onnx_input)}")
print(f"Sample input: {onnx_input}")
ort_session = onnxruntime.InferenceSession("model_avg.onnx", providers=['CPUExecutionProvider'])
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)}
onnxruntime_outputs = ort_session.run(None, onnxruntime_input)
torch_outputs = net(move_input, mask_input)
torch_outputs = onnx_program.adapt_torch_outputs_to_onnx(torch_outputs)
assert len(torch_outputs) == len(onnxruntime_outputs)
for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
    torch.testing.assert_close(torch_output, torch.tensor(onnxruntime_output))

print("PyTorch and ONNX Runtime output matched!")
print(f"Output length: {len(onnxruntime_outputs)}")
print(f"Sample output: {onnxruntime_outputs}")


net = modelTime().to('cpu')

state_dict = torch.load('model_avg_time.pth', map_location='cpu')
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    k = k.replace('module.', '')
    new_state_dict[k] = v

net.load_state_dict(new_state_dict)
net.eval()

move_input = torch.from_numpy(np.random.randint(0, 1+NUMBER_OF_MOVES, size=(1, 150), dtype=np.int64)).to(torch.int64)
mask_input = torch.from_numpy(np.random.randint(0, 2, size=(1, 150), dtype=np.bool_)).to(torch.bool)
time_base_input = torch.randn(1, dtype=torch.float32)
time_bonus_input = torch.randn(1, dtype=torch.float32)
onnx_program = torch.onnx.dynamo_export(net, move_input, mask_input, time_base_input, time_bonus_input)
onnx_program.save("model_avg_time.onnx")

onnx_model = onnx.load("model_avg_time.onnx")
onnx.checker.check_model(onnx_model)
print('Successfully saved time model')
onnx_input = onnx_program.adapt_torch_inputs_to_onnx(move_input, mask_input, time_base_input, time_bonus_input)
print(f"Input length: {len(onnx_input)}")
print(f"Sample input: {onnx_input}")
ort_session = onnxruntime.InferenceSession("model_avg_time.onnx", providers=['CPUExecutionProvider'])
onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)}
onnxruntime_outputs = ort_session.run(None, onnxruntime_input)
torch_outputs = net(move_input, mask_input, time_base_input, time_bonus_input)
torch_outputs = onnx_program.adapt_torch_outputs_to_onnx(torch_outputs)
assert len(torch_outputs) == len(onnxruntime_outputs)
for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
    torch.testing.assert_close(torch_output, torch.tensor(onnxruntime_output))

print("PyTorch and ONNX Runtime output matched!")
print(f"Output length: {len(onnxruntime_outputs)}")
print(f"Sample output: {onnxruntime_outputs}")
