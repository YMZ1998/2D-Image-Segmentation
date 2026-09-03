import argparse

import torch

# from network.UDTransNet.ETransUNet import ETransUNet
# from network.UDTransNet.UDTransNet import UDTransNet
from network.UNet import UNet
from network.efficientnet_unet import EfficientUNet
# from network.mobilenet_unet import MobileV3UNet


def get_device(requested="cuda"):
    device = torch.device(requested if requested.startswith("cuda") and torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    return device


def get_best_weight_path(args, verbose=True):
    weights_path = "save_weights/{}_{}_best_model.pth".format(args.arch, args.deep_supervision)
    if verbose:
        print("best weight: ", weights_path)
    return weights_path


def get_latest_weight_path(args, verbose=False):
    weights_path = "save_weights/{}_{}_latest_model.pth".format(args.arch, args.deep_supervision)
    if verbose:
        print("latest weight: ", weights_path)
    return weights_path


efficientnet_dict = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                     'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                     'efficientnet_b6', 'efficientnet_b7', 'efficientnet_v2_s']


def get_model(args, is_convert_onnx=False, pretrain_backbone=True):
    print('★'*30)
    print(f'model:{args.arch}\n'
          f'epoch:{args.epochs}\n'
          f'batch size:{args.batch_size}\n'
          f'image size:{args.image_size}')
    print('★'*30)
    device = get_device(args.device)
    if args.arch == 'unet':
        model = UNet(in_channels=args.in_channels, num_classes=args.num_classes, base_c=32).to(device)
    # elif args.arch == 'mobilenet':
    #     model = MobileV3UNet(num_classes=args.num_classes, pretrain_backbone=True).to(device)
    elif args.arch == 'efficientnet' or args.arch in efficientnet_dict:
        model = EfficientUNet(num_classes=args.num_classes, pretrain_backbone=pretrain_backbone,
                              model_name=args.arch, deep_supervision=args.deep_supervision,
                              is_convert_onnx=is_convert_onnx).to(device)
    # if args.arch == 'efficientnet2':
    #     from efficientunet import get_efficientunet_b1
    #     model = get_efficientunet_b1(out_channels=args.num_classes, concat_input=True, pretrained=True).to(device)
    # elif args.arch == 'UDTransNet':
    #     model = UDTransNet(n_channels=args.in_channels, n_classes=args.num_classes, img_size=args.image_size).to(device)
    # elif args.arch == 'ETransUNet':
    #     model = ETransUNet(n_channels=args.in_channels, n_classes=args.num_classes, img_size=args.image_size).to(device)
    else:
        raise ValueError('arch error')
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="pytorch training")
    parser.add_argument('--arch', '-a', metavar='ARCH', default='efficientnet_b1',
                        help='unet/mobilenet/efficientnet_b1/efficientnet_v2_s/UDTransNet/ETransUNet')
    parser.add_argument("--data_path", default="data/dataset",
                        help="prepared dataset root containing train/ and test/")
    parser.add_argument("--augmented_path", default="data/augmented",
                        help="augmented dataset root used to prepare train/test data")
    parser.add_argument("--num_classes", default=4, type=int)
    parser.add_argument("--in_channels", default=1, type=int)
    parser.add_argument("--image_size", default=704, type=int)
    parser.add_argument("--test_ratio", default=0.2, type=float)
    parser.add_argument("--split_seed", default=42, type=int)
    parser.add_argument("--skip_data_prepare", action="store_true",
                        help="use an already prepared data_path without rebuilding it")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch_size", default=4, type=int)
    parser.add_argument("--epochs", default=300, type=int, metavar="N",
                        help="number of total epochs to train")
    # Optimizer options
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--resume', default=0, type=int, help='resume from checkpoint')
    parser.add_argument('--deep_supervision', default=0, help='deep supervision training')
    parser.add_argument('--multi_scale', default=0, help='multi-scale training')
    parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save_best', default=True, type=bool, help='only save best metric weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    model = get_model(args)
    print(model)
