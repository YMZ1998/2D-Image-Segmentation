import os
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def loss_plot(args, train_loss, val_loss, plot_save_path='log/plot/'):
    """
    Plot and save the training and validation loss.

    Parameters:
        args: Argument parser object containing 'arch' and 'epochs' attributes.
        train_loss: List of training loss values.
        val_loss: List of validation loss values.
        plot_save_path: Directory where the plot will be saved.
    """
    num = len(train_loss)
    x = list(range(1, num + 1))
    ensure_dir_exists(plot_save_path)

    save_loss = os.path.join(plot_save_path, f"{args.arch}_{args.epochs}_loss.png")
    plt.figure()
    plt.xlim([1, num])
    plt.ylim([np.min(val_loss) / 2, np.max(val_loss) + 0.1])
    plt.plot(x, train_loss, 'r', label='train loss')
    plt.plot(x, val_loss, 'g', label='val loss')
    plt.legend()
    plt.savefig(save_loss, dpi=300)
    plt.close()


def metrics_plot(args, name, metrics_value, plot_save_path='log/plot/'):
    """
    Plot and save the specified metric.

    Parameters:
        args: Argument parser object containing 'arch' and 'epochs' attributes.
        name: Name of the metric.
        metrics_value: List of metric values.
        plot_save_path: Directory where the plot will be saved.
    """
    num = len(metrics_value)
    x = list(range(1, num + 1))
    ensure_dir_exists(plot_save_path)

    save_metrics = os.path.join(plot_save_path, f"{args.arch}_{args.epochs}_{name}.png")
    plt.figure()
    plt.plot(x, metrics_value, 'b', label=name)
    plt.xlim([1, num])
    plt.ylim([np.min(metrics_value) - 0.01, np.max(metrics_value) + 0.01])
    plt.legend()
    plt.savefig(save_metrics, dpi=300)
    plt.close()


if __name__ == '__main__':
    from parse_args import parse_args

    args = parse_args()
    args.epochs = 5

    train_loss = [0, 0.6, 0.8, 0.8, 0.92]
    val_loss = [0, 0.3, 0.7, 0.86, 0.97]
    dice_scores = [0.82, 0.86, 0.88, 0.92, 0.95]

    loss_plot(args, train_loss, val_loss)
    metrics_plot(args, "dice", dice_scores)
