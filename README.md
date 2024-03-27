# HOG + SVM 行人检测模型

基于 Python + OpenCV 搭建的行人检测训练模型项目。HOG 特征的简要介绍可参考：[https://zhuanlan.zhihu.com/p/40960756](https://zhuanlan.zhihu.com/p/40960756)

## 文件目录说明

.\data: 以 npy 格式存储的训练/测试数据集（可自定义）；

.\INRIAData: INRIA 行人检测数据集（预处理过）；

.\model: 训练后的模型存储路径；

.\test-xxx: 用于模型实战检测；

dataset.py: 数据集预处理函数；

eval.py: 训练配置函数；

metrics.py: 评估指标函数；

test.py: 可以按提示选择自己的图片进行行人检测，估计模型效果；

train.py: 执行训练过程；

utils.py: 封装的工具函数；

## 使用说明

### 训练模型

.\model 目录下已经存在训练好的 3 个模型，可直接调用。如果你需要重新训练，请切换到项目根目录，并在终端输入：

```
python train.py --model svc-rbf --useown .\mydataset

python train.py --model ? --useown ?
--model: 选择模型类型，默认选择 svc-rbf
参数： svc-linear | svc-poly | svc-rbf
--useown: 是否使用自己的数据集进行训练，若使用请添加数据集的路径，并按照 .\data\train .\data\test 的格式存放数据
```

### 评估指标

运行 metrics.py 可查看指定模型在测试集上的混淆矩阵，准确率，精确率和召回率。

### 测试模型

指定好测试图片的路径后，运行 test.py 可生成行人检测框选后的图片，具体请查看 test.py 文件。 
