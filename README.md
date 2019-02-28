**建议运行在pytorch0.4.0环境中**
##Prerequisites
* Linux
* Python 3+
* opencv
* pytorch 0.4.0+
* tensorboardX (pip install tensorboardX tensorflow)
* CUDA cuDNN (you should better train with a gpu)


## TRAIN
```bash
python train_point_regression.py
```
* 生成的模型保存在./checkpoints 中

## EVAL
```bash
python eval_point_regression.py
```
* 在主函数中设定模型的路径，输出模型在测试集上的像素误差

## datasets.py
* BoxPointFromCPM   根据cpm网络的输出和gt生成数据
* BoxPointFromSeven 根据gt生成数据随机偏移数据（目前设定为96x96的crop image，如需修改，需在代码中手动修改）


## networks.py
* MobileNet         point_regression 使用mobilenet1.0 width = 0.75，并且剪掉了两层conv_dw，速度相比原版加快40%


## pytorch2onnx.py
* 将pytorch转换为onnx模型，暂时用不上