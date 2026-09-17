import os

import numpy as np
import onnx
import onnxruntime
import torch
from torch import nn


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


class OnnxModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model(image)["out"]


def convert_onnx():

    from parse_args import parse_args, get_model, get_best_weight_path

    args = parse_args()

    # ============================================================
    # 1. 创建模型
    # ============================================================

    model = get_model(
        args,
        pretrain_backbone=False
    )

    weights_path = get_best_weight_path(args)

    device = torch.device("cpu")

    print("Using device:", device)
    print("PyTorch:", torch.__version__)

    # ============================================================
    # 2. 加载权重
    # ============================================================

    checkpoint = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=False
    )

    state = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state)

    model = OnnxModel(model)
    model = model.to(device)
    model.eval()

    # ============================================================
    # 3. 创建输入
    #
    # [N, C, H, W]
    # [1, 1, H, W]
    # ============================================================

    batch_size = 1
    image_size = args.image_size

    x = torch.rand(
        batch_size,
        1,
        image_size,
        image_size,
        dtype=torch.float32,
        device=device
    )

    print()
    print("=" * 60)
    print("Input shape :", tuple(x.shape))
    print("=" * 60)

    # ============================================================
    # 4. PyTorch 推理
    # ============================================================

    with torch.no_grad():
        torch_out = model(x)

    print("PyTorch output shape:", tuple(torch_out.shape))

    # ============================================================
    # 5. ONNX 文件
    # ============================================================

    os.makedirs("save_weights", exist_ok=True)

    onnx_file_name = (
        f"save_weights/{args.arch}_best_model.onnx"
    )

    # ============================================================
    # 6. ONNX Export
    #
    # 关键：
    #   dynamo=False
    #
    # TensorRT 8.6 建议使用传统 exporter
    # ============================================================

    print()
    print("Exporting ONNX...")

    with torch.no_grad():

        torch.onnx.export(
            model,
            x,
            onnx_file_name,

            input_names=["input"],
            output_names=["output"],

            # TensorRT 8.6 建议
            opset_version=17,

            # 强制使用传统 exporter
            dynamo=False,

            # 不需要动态 batch
            dynamic_axes=None,

            # 不生成 external data
            external_data=False,

            verbose=False
        )

    print("ONNX saved:", onnx_file_name)

    # ============================================================
    # 7. 检查 ONNX
    # ============================================================

    print()
    print("Checking ONNX...")

    onnx_model = onnx.load(onnx_file_name)

    onnx.checker.check_model(onnx_model)

    print("ONNX checker: OK")

    # ============================================================
    # 8. 打印 ONNX 输入输出
    # ============================================================

    print()
    print("=" * 60)
    print("ONNX IO")
    print("=" * 60)

    for value in onnx_model.graph.input:

        shape = []

        for dim in value.type.tensor_type.shape.dim:

            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            else:
                shape.append("?")

        print(
            "Input :",
            value.name,
            shape
        )

    for value in onnx_model.graph.output:

        shape = []

        for dim in value.type.tensor_type.shape.dim:

            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            else:
                shape.append("?")

        print(
            "Output:",
            value.name,
            shape
        )

    # ============================================================
    # 9. ONNX Runtime 验证
    # ============================================================

    print()
    print("Testing with ONNX Runtime...")

    ort_session = onnxruntime.InferenceSession(
        onnx_file_name,
        providers=["CPUExecutionProvider"]
    )

    ort_inputs = {
        "input": to_numpy(x)
    }

    ort_outs = ort_session.run(
        ["output"],
        ort_inputs
    )

    ort_output = ort_outs[0]

    print(
        "ONNX Runtime output shape:",
        ort_output.shape
    )

    # ============================================================
    # 10. PyTorch / ONNX 数值对比
    # ============================================================

    torch_output = to_numpy(torch_out)

    print()
    print("=" * 60)
    print("PyTorch vs ONNX")
    print("=" * 60)

    print(
        "PyTorch shape:",
        torch_output.shape
    )

    print(
        "ONNX shape   :",
        ort_output.shape
    )

    max_diff = np.max(
        np.abs(torch_output - ort_output)
    )

    mean_diff = np.mean(
        np.abs(torch_output - ort_output)
    )

    print("Max diff :", max_diff)
    print("Mean diff:", mean_diff)

    np.testing.assert_allclose(
        torch_output,
        ort_output,
        rtol=1e-3,
        atol=1e-3
    )

    print()
    print("ONNX Runtime verification: PASSED")
    print()
    print("Export completed:")
    print(onnx_file_name)


if __name__ == "__main__":
    convert_onnx()