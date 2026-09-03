import numpy as np
import onnx
import onnxruntime
import torch


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def convert_onnx():
    from parse_args import parse_args, get_model, get_best_weight_path

    args = parse_args()
    # create model
    model = get_model(args, is_convert_onnx=True)
    weights_path = get_best_weight_path(args)

    device = torch.device("cpu")
    print("using {} device.".format(device))

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)
    onnx_file_name = "save_weights/{}_best_model.onnx".format(args.arch)
    batch_size = 1

    model.eval()
    # input to the model
    # [batch, channel, height, width]
    x = torch.rand(batch_size, args.image_size, args.image_size, 3, requires_grad=True)
    torch_out = model(x)
    # print(torch_out)
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      onnx_file_name,  # where to save the model (can be a file or file-like object)
                      input_names=["input"],
                      output_names=["output"],
                      external_data=False,
                      verbose=True)

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
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    print(f"save onnx model to {onnx_file_name}.")


if __name__ == '__main__':
    convert_onnx()
