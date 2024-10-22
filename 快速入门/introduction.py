import mindspore
from mindspore import nn
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset
# Download data from open datasets
from download import download

# # 数据集下载
# url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
#       "notebook/datasets/MNIST_Data.zip"
# path = download(url, "./", kind="zip", replace=True)
train_dataset = MnistDataset('MNIST_Data/train')
test_dataset = MnistDataset('MNIST_Data/test')
print(train_dataset.get_col_names())

# 对输入的dataset进行图像预处理和批量化：
"""
    对图像进行预处理：
    对图像进行缩放，将图像像素值缩放到[0, 1]区间；
    对图像进行标准化处理；
    改变图像通道顺序从HWC到CHW；
    将标签转换为mindspore.int32类型；
    将数据集按指定批量大小batch_size进行批量化
"""


def datapipe(dataset, batch_size):
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
    label_transform = transforms.TypeCast(mindspore.int32)

    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset


# Map vision transforms and batch dataset
train_dataset = datapipe(train_dataset, 64)
test_dataset = datapipe(test_dataset, 64)
for image, label in test_dataset.create_tuple_iterator():
    print(f"Shape of image [N, C, H, W]: {image.shape} {image.dtype}")
    print(f"Shape of label: {label.shape} {label.dtype}")
    break
for data in test_dataset.create_dict_iterator():
    print(f"Shape of image [N, C, H, W]: {data['image'].shape} {data['image'].dtype}")
    print(f"Shape of label: {data['label'].shape} {data['label'].dtype}")
    break


# 网络构建
class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28 * 28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10)
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits


model = Network()
print(model)
# 定义交叉熵损失函数和随机梯度下降（SGD）优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = nn.SGD(model.trainable_params(), 1e-2)


# 定义前向传播函数
def forward_fn(data, label):
    """
    前向传播函数，计算模型输出和损失。

    参数:
        data: 输入数据。
        label: 标签数据。

    返回:
        loss: 损失值。
        logits: 模型输出 logits。
    """
    logits = model(data)
    loss = loss_fn(logits, label)
    return loss, logits


# 创建计算梯度的函数
grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)


# 定义训练步函数
def train_step(data, label):
    """
    训练步函数，执行一步训练并更新模型参数。

    参数:
        data: 输入数据。
        label: 标签数据。

    返回:
        loss: 损失值。
    """
    (loss, _), grads = grad_fn(data, label)
    optimizer(grads)
    return loss


# 定义训练函数
def train(model, dataset):
    """
    训练函数，对整个数据集进行多次训练。

    参数:
        model: 待训练的模型。
        dataset: 训练数据集。
    """
    size = dataset.get_dataset_size()
    model.set_train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)

        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")


# 定义测试函数
def test(model, dataset, loss_fn):
    """
    测试函数，评估模型在测试数据集上的性能。

    参数:
        model: 待测试的模型。
        dataset: 测试数据集。
        loss_fn: 损失函数。
    """
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred = model(data)
        total += len(data)
        test_loss += loss_fn(pred, label).asnumpy()
        correct += (pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    print(f"Test: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# 定义训练和测试的轮数
epochs = 3
# 循环进行训练和测试
for t in range(epochs):
    # 打印当前轮次的信息，用于跟踪训练进度
    print(f"Epoch {t + 1}\n-------------------------------")
    # 调用train函数进行模型训练
    train(model, train_dataset)
    # 调用test函数进行模型测试，并打印测试结果
    test(model, test_dataset, loss_fn)
# 打印训练完成的信息
print("Done!")


# 保存模型参数到文件
mindspore.save_checkpoint(model, "model.ckpt")
print("Saved Model to model.ckpt")

# 实例化网络模型对象
model = Network()

# 从文件中加载模型参数
param_dict = mindspore.load_checkpoint("model.ckpt")

# 将加载的参数字典中的参数加载到模型中，返回未加载的参数信息
param_not_load, _ = mindspore.load_param_into_net(model, param_dict)
print(param_not_load)

# 设置模型为评估模式
model.set_train(False)

# 遍历测试数据集，进行预测
for data, label in test_dataset:
    # 输入数据到模型，得到预测结果
    pred = model(data)
    # 获取预测结果中概率最大的类别索引
    predicted = pred.argmax(1)
    # 打印预测结果和真实标签
    print(f'Predicted: "{predicted[:10]}", Actual: "{label[:10]}"')
    # 只需打印一次预测结果，预测完第一个batch后即退出循环
    break

