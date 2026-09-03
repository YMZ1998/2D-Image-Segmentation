import glob
import os
import shutil
import time

import cv2
import numpy as np
import onnxruntime
import tqdm

from parse_args import parse_args, get_model

# Load ONNX model
args = parse_args()
get_model(args)
onnx_file_name = "../save_weights/{}_best_model.onnx".format(args.arch)
session = onnxruntime.InferenceSession(onnx_file_name)
start_time = time.time()
paths = glob.glob(args.data_path+"/test/image/*.*")[0:-1]
# paths = glob.glob("/mnt/algo_storage_server/UNet/Dataset/implant2/*.*")[::3]
# paths = glob.glob(args.data_path + "/augmentation_test/image/*.*")[::20]

result_path = './onnx'
# result_path = '/mnt/algo_storage_server/UNet/Dataset/data/onnx—predict/'

if os.path.exists(result_path):
    shutil.rmtree(result_path)
os.mkdir(result_path)

# Load golden images
for path in tqdm.tqdm(paths):
    # if 'image' not in path and 'implant' in path:
    #     continue
    # Compute ONNX model output
    origin_image = cv2.imread(path)
    if origin_image is None:
        print("{} is None".format(path))
        continue
    image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.image_size, args.image_size), interpolation=cv2.INTER_CUBIC)
    image = np.array(image, np.float32)
    image = image / 127.5 - 1
    image = np.expand_dims(image, 0)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: image})

    # Compare ONNX model output with golden image
    output_image = result[0].argmax(3).squeeze(0)
    output_image[output_image == 1] = 64
    output_image[output_image == 2] = 129
    output_image[output_image == 3] = 192
    output_image[output_image == 4] = 255
    prediction = output_image.astype(np.uint8)

    predict_result = cv2.resize(prediction, (origin_image.shape[1], origin_image.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(result_path, os.path.splitext(os.path.basename(path))[0] + "_origin.png"), origin_image)
    cv2.imwrite(os.path.join(result_path, os.path.splitext(os.path.basename(path))[0] + "_predict.png"), predict_result)

total_time = time.time() - start_time
print("time {}s, fps {}".format(total_time, len(paths) / total_time))
