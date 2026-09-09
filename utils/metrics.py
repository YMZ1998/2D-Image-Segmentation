import numpy as np

def compute_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask)
    union = np.logical_or(pred_mask, true_mask)
    iou = (np.sum(intersection) + 1e-6) / (np.sum(union) + 1e-6)
    return iou


def compute_mean_iou(pred_images, true_images):
    num_classes = 3  # 获取类别数
    pred_images = np.eye(num_classes)[pred_images]  # 进行One-Hot编码
    true_images = np.eye(num_classes)[true_images]  # 进行One-Hot编码
    # print(pred_images.shape,true_images.shape)
    mean_iou = 0.0

    for i in range(num_classes):
        pred_image = pred_images[:, :, i]
        true_image = true_images[:, :, i]
        iou = compute_iou(pred_image, true_image)
        mean_iou += iou

    mean_iou /= num_classes
    return mean_iou


def image_pair_to_confusion_matrix(pred_img, true_img, num_classes):
    # 将预测图像和真实图像转换为标签矩阵
    pred_labels = pred_img.flatten()
    true_labels = true_img.flatten()

    # 初始化混淆矩阵
    confusion_matrix = np.zeros((num_classes, num_classes))

    # 计算混淆矩阵
    for i in range(len(pred_labels)):
        pred_label = int(pred_labels[i])
        true_label = int(true_labels[i])
        confusion_matrix[true_label, pred_label] += 1

    return confusion_matrix


def calculate_iou(confusion_matrix):
    # 计算每个类别的IoU
    ious = []
    for i in range(len(confusion_matrix)):
        true_positive = confusion_matrix[i, i]
        false_positive = np.sum(confusion_matrix[:, i]) - true_positive
        false_negative = np.sum(confusion_matrix[i, :]) - true_positive
        iou = (true_positive + 1e-6) / (true_positive + false_positive + false_negative + 1e-6)
        ious.append(iou)
    return ious


def calculate_miou(confusion_matrix):
    ious = calculate_iou(confusion_matrix)
    miou = np.mean(ious)
    return miou


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
        # 创建混淆矩阵
            self.mat = np.zeros((n, n), dtype=np.int64)
        # 寻找GT中为目标的像素索引
        k = (a >= 0) & (a < n)
        # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
        inds = n * a[k] + b[k]
        self.mat += np.bincount(inds, minlength=n ** 2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat = None

    def compute(self):
        h = self.mat
        # 计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)
        acc_global = np.diag(h).sum() / h.sum()
        # 计算每个类别的准确率
        acc = np.diag(h) / h.sum(1)
        # 计算每个类别预测与真实目标的iou
        iu = np.diag(h) / (h.sum(1) + h.sum(0) - np.diag(h))
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not np.distributed.is_available():
            return
        if not np.distributed.is_initialized():
            return
        np.distributed.barrier()
        np.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.2f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.2f}').format(
            acc_global.item() * 100,
            ['{:.2f}'.format(i) for i in (acc * 100).tolist()],
            ['{:.2f}'.format(i) for i in (iu * 100).tolist()],
            iu.mean().item() * 100)

    def get_miou(self):
        acc_global, acc, iu = self.compute()
        return iu.mean().item()


if __name__ == '__main__':
    pass
