import numpy as np
import onnx
import onnxruntime
import torch
from torch import nn


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class OnnxModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model(image)["out"]


def convert_onnx():
    from parse_args import parse_args, get_model, get_best_weight_path

    args = parse_args()
    # create model
    model = get_model(args, pretrain_backbone=False)
    weights_path = get_best_weight_path(args)

    device = torch.device("cpu")
    print("using {} device.".format(device))

    # load weights
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    model = OnnxModel(model).to(device).eval()
    onnx_file_name = "save_weights/{}_best_model.onnx".format(args.arch)
    batch_size = 1

    # NCHW grayscale input, identical to training and Python inference.
    x = torch.rand(batch_size, 1, args.image_size, args.image_size)
    torch_out = model(x)
    # print(torch_out)
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      onnx_file_name,  # where to save the model (can be a file or file-like object)
                      input_names=["input"],
                      output_names=["output"],
                      opset_version=17,
                      external_data=False,
                      verbose=False)

    # check the onnx model
    onnx_model = onnx.load(onnx_file_name)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(onnx_file_name)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    # print(ort_inputs['input'].shape)
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and Pytorch results
    # assert_allclose: Raises an AssertionError if two objects are not equal up to desired tolerance.
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-03)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    print(f"save onnx model to {onnx_file_name}.")


if __name__ == '__main__':
    convert_onnx()
