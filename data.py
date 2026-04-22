# 使用torchvision.datasets加载CIFAR-10
 
 
# 训练集加载
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True,
    download=True,
    transform=transform
)
 
# 测试集加载
test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
