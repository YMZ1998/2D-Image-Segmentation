import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from inference_utils import (
    clean_circular_roi,
    colorize_mask,
    overlay_prediction,
)
from segmentation_config import CLASS_NAMES, IMAGE_SIZE, ROI_RADIUS_RATIO

TRT_DTYPE_TO_NUMPY = {
    "FLOAT": np.float32,
    "HALF": np.float16,
    "INT8": np.int8,
    "INT32": np.int32,
    "BOOL": np.bool_,
}


def newest_engine(directory: Path = Path("../save_weights")) -> Path:
    candidates = []
    for pattern in ("*.engine", "*.plan", "*.trt"):
        candidates.extend(directory.glob(pattern))
    candidates = [
        path
        for path in candidates
        if not path.name.endswith(".timing.cache")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No TensorRT engine found in {directory} (*.engine, *.plan, *.trt)"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def newest_onnx(directory: Path = Path("save_weights")) -> Path:
    candidates = list(directory.glob("*.onnx"))
    if not candidates:
        raise FileNotFoundError(f"No ONNX model found in {directory}/*.onnx")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def matching_onnx_for_engine(engine_path: Path) -> Path:
    onnx_path = engine_path.with_suffix(".onnx")
    if onnx_path.is_file():
        return onnx_path
    return newest_onnx(PROJECT_ROOT / "save_weights")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-image OCT segmentation with a TensorRT engine."
    )
    parser.add_argument("--image", type=Path, default=r'../../input.png')
    parser.add_argument(
        "--engine",
        type=Path,
        default=r'D:\Code\2D-Image-Segmentation\save_weights/efficientnet_b1_best_model.engine',
        help="TensorRT serialized engine path; defaults to newest save_weights/*.engine",
    )
    parser.add_argument("--onnx", type=Path, help="ONNX path used to rebuild an incompatible engine")
    parser.add_argument("--fp16", action="store_true", help="Rebuild TensorRT engine with FP16 when supported")
    parser.add_argument("--workspace-gb", type=float, default=6.0, help="TensorRT build workspace in GB")
    parser.add_argument(
        "--no-rebuild-engine",
        action="store_true",
        help="Do not rebuild from ONNX when engine deserialization fails",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("predictions_trt"))
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--roi-radius-ratio", type=float, default=ROI_RADIUS_RATIO)
    parser.add_argument("--keep-border-info", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def import_trt_runtime():
    try:
        import tensorrt as trt
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
    except ImportError as error:
        raise RuntimeError(
            "TensorRT 推理需要安装 tensorrt 和 pycuda，并在支持 CUDA 的环境中运行。"
        ) from error
    return trt, cuda


def load_engine(engine_path: Path, trt):
    total_start = time.perf_counter()

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    start = time.perf_counter()
    with engine_path.open("rb") as file:
        engine_data = file.read()
    read_time = time.perf_counter() - start

    start = time.perf_counter()
    engine = runtime.deserialize_cuda_engine(engine_data)
    deserialize_time = time.perf_counter() - start

    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

    total_time = time.perf_counter() - total_start

    print(f"Engine 文件读取: {read_time:.3f} s")
    print(f"Engine 反序列化: {deserialize_time:.3f} s")
    print(f"Engine 总加载耗时: {total_time:.3f} s")

    return engine


def build_engine_from_onnx(
        onnx_path: Path,
        engine_path: Path,
        trt,
        fp16: bool = False,
        workspace_gb: float = 2.0,
):
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found for TensorRT rebuild: {onnx_path}")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flags)
    else:
        # TensorRT 10+ uses explicit batch by default and removed this flag.
        network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    with onnx_path.open("rb") as file:
        parsed = parser.parse(file.read())
    if not parsed:
        messages = []
        for index in range(parser.num_errors):
            messages.append(str(parser.get_error(index)))
        raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(messages))

    config = builder.create_builder_config()
    workspace_bytes = int(max(0.25, workspace_gb) * (1 << 30))
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    print(f"Rebuilding TensorRT engine from ONNX: {onnx_path}")
    print(f"Target engine: {engine_path}")
    if hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT build_serialized_network failed")
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.write_bytes(bytes(serialized))
        return load_engine(engine_path, trt)

    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("TensorRT build_engine failed")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(engine.serialize()))
    return engine


def binding_names(engine) -> list[str]:
    if hasattr(engine, "num_io_tensors"):
        return [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
    return [engine.get_binding_name(index) for index in range(engine.num_bindings)]


def is_input(engine, name: str) -> bool:
    if hasattr(engine, "get_tensor_mode"):
        import tensorrt as trt

        return engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
    return engine.binding_is_input(engine.get_binding_index(name))


def tensor_shape(engine, context, name: str) -> tuple[int, ...]:
    if hasattr(engine, "get_tensor_shape"):
        shape = tuple(context.get_tensor_shape(name))
        if any(dim < 0 for dim in shape):
            shape = tuple(engine.get_tensor_shape(name))
        return shape
    return tuple(context.get_binding_shape(engine.get_binding_index(name)))


def tensor_dtype(engine, name: str):
    if hasattr(engine, "get_tensor_dtype"):
        dtype = engine.get_tensor_dtype(name)
    else:
        dtype = engine.get_binding_dtype(engine.get_binding_index(name))
    return TRT_DTYPE_TO_NUMPY.get(str(dtype).split(".")[-1], np.float32)


def set_input_shape_if_dynamic(engine, context, name: str, shape: tuple[int, ...]) -> None:
    if hasattr(context, "set_input_shape"):
        current = tuple(engine.get_tensor_shape(name))
        if any(dim < 0 for dim in current):
            context.set_input_shape(name, shape)
        return

    index = engine.get_binding_index(name)
    current = tuple(engine.get_binding_shape(index))
    if any(dim < 0 for dim in current):
        context.set_binding_shape(index, shape)


def infer_layout_and_size(shape: tuple[int, ...], fallback_size: int) -> tuple[str, int, int, int]:
    if len(shape) != 4:
        raise ValueError(f"Expected a 4D TensorRT input, got: {shape}")
    if shape[1] in (1, 3):
        return "NCHW", int(shape[1]), int(shape[2]), int(shape[3])
    if shape[3] in (1, 3):
        return "NHWC", int(shape[3]), int(shape[1]), int(shape[2])
    return "NCHW", 1, fallback_size, fallback_size


def prepare_input(gray: np.ndarray, shape: tuple[int, ...], fallback_size: int) -> tuple[
    np.ndarray, str, tuple[int, int]]:
    shape = tuple(fallback_size if dim < 0 else dim for dim in shape)
    layout, channels, height, width = infer_layout_and_size(shape, fallback_size)
    resized = np.asarray(
        Image.fromarray(gray).resize((width, height), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    resized = resized / 127.5 - 1.0
    resized = np.repeat(resized[..., None], channels, axis=2)
    tensor = resized.transpose(2, 0, 1)[None] if layout == "NCHW" else resized[None]
    return np.ascontiguousarray(tensor, dtype=np.float32), layout, (height, width)


def output_to_mask(output: np.ndarray) -> np.ndarray:
    output = np.asarray(output)
    if output.ndim == 4:
        valid_class_counts = range(2, len(CLASS_NAMES) + 1)
        if output.shape[1] in valid_class_counts:
            return output.argmax(axis=1)[0].astype(np.uint8)
        if output.shape[-1] in valid_class_counts:
            return output.argmax(axis=-1)[0].astype(np.uint8)
        raise ValueError(f"Cannot find class axis in TensorRT output shape: {output.shape}")
    if output.ndim == 3 and output.shape[0] == 1:
        return output[0].astype(np.uint8)
    if output.ndim == 2:
        return output.astype(np.uint8)
    raise ValueError(f"Unsupported TensorRT output shape: {output.shape}")


def execute(engine, context, cuda, input_name: str, output_name: str, tensor: np.ndarray):
    stream = cuda.Stream()
    input_shape = tuple(tensor.shape)
    set_input_shape_if_dynamic(engine, context, input_name, input_shape)
    output_shape = tensor_shape(engine, context, output_name)
    if any(dim < 0 for dim in output_shape):
        raise ValueError(f"Output shape is still dynamic after setting input shape: {output_shape}")

    output = np.empty(output_shape, dtype=tensor_dtype(engine, output_name))
    device_input = cuda.mem_alloc(tensor.nbytes)
    device_output = cuda.mem_alloc(output.nbytes)

    cuda.memcpy_htod_async(device_input, tensor, stream)

    start = time.perf_counter()
    if hasattr(context, "set_tensor_address"):
        context.set_tensor_address(input_name, int(device_input))
        context.set_tensor_address(output_name, int(device_output))
        context.execute_async_v3(stream_handle=stream.handle)
    else:
        bindings = [0] * engine.num_bindings
        bindings[engine.get_binding_index(input_name)] = int(device_input)
        bindings[engine.get_binding_index(output_name)] = int(device_output)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(output, device_output, stream)
    stream.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return output, elapsed_ms


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    engine_path = args.engine or newest_engine()
    trt, cuda = import_trt_runtime()
    try:
        if not engine_path.is_file():
            onnx_path = args.onnx or matching_onnx_for_engine(engine_path)
            engine = build_engine_from_onnx(
                onnx_path=onnx_path,
                engine_path=engine_path,
                trt=trt,
                fp16=args.fp16,
                workspace_gb=args.workspace_gb,
            )
        else:
            engine = load_engine(engine_path, trt)
    except RuntimeError as error:
        if args.no_rebuild_engine:
            raise
        print(str(error))
        print(
            "当前 TensorRT engine 无法在本环境反序列化，"
            "这通常是 TensorRT/CUDA/GPU 版本不匹配导致的。"
        )
        onnx_path = args.onnx or matching_onnx_for_engine(engine_path)
        engine = build_engine_from_onnx(
            onnx_path=onnx_path,
            engine_path=engine_path,
            trt=trt,
            fp16=args.fp16,
            workspace_gb=args.workspace_gb,
        )
    context = engine.create_execution_context()

    names = binding_names(engine)
    input_names = [name for name in names if is_input(engine, name)]
    output_names = [name for name in names if not is_input(engine, name)]
    if len(input_names) != 1 or len(output_names) < 1:
        raise RuntimeError(f"Expected one input and at least one output, got {input_names}, {output_names}")
    input_name = input_names[0]
    output_name = output_names[0]

    source = Image.open(args.image).convert("L")
    gray = np.asarray(source)
    if not args.keep_border_info:
        gray = clean_circular_roi(gray, args.roi_radius_ratio)

    input_shape = tuple(
        dim if isinstance(dim, int) and dim > 0 else args.image_size
        for dim in tensor_shape(engine, context, input_name)
    )
    tensor, layout, inference_size = prepare_input(gray, input_shape, args.image_size)
    output, elapsed_ms = execute(engine, context, cuda, input_name, output_name, tensor)

    prediction = output_to_mask(output)
    prediction = np.asarray(
        Image.fromarray(prediction).resize(source.size, Image.Resampling.NEAREST)
    )
    color_mask = colorize_mask(prediction)
    gray_rgb = np.repeat(gray[..., None], 3, axis=2)
    overlay = overlay_prediction(gray_rgb, prediction, args.alpha)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "mask": args.output_dir / f"{args.image.stem}_mask.png",
        "color": args.output_dir / f"{args.image.stem}_color.png",
        "overlay": args.output_dir / f"{args.image.stem}_overlay.png",
    }
    Image.fromarray(prediction).save(paths["mask"])
    Image.fromarray(color_mask).save(paths["color"])
    Image.fromarray(overlay).save(paths["overlay"])

    present = [CLASS_NAMES[index] for index in np.unique(prediction) if index != 0]
    print(f"engine: {engine_path}")
    print(f"input: {input_name} {tuple(tensor.shape)} ({layout}, inference={inference_size})")
    print(f"output: {output_name} {tuple(output.shape)}")
    print(f"inference: {elapsed_ms:.2f} ms")
    print(f"predicted classes: {', '.join(present) if present else 'background only'}")
    for name, path in paths.items():
        print(f"{name}: {path}")

    if not args.no_show:
        figure, axes = plt.subplots(1, 3, figsize=(16, 6))
        axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("Input (grayscale)")
        axes[1].imshow(color_mask)
        axes[1].set_title("TensorRT prediction")
        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay (alpha={args.alpha:.2f})")
        for axis in axes:
            axis.axis("off")
        figure.suptitle(f"{engine_path.name} | {elapsed_ms:.1f} ms")
        figure.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
