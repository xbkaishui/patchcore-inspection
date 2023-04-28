# 修改patchcore 代码，主要是修复了一些bug和增加预测接口实现

## 如何训练
参考run_train.sh 脚本

## 如何预测
运行 bin/predict_image.py
需要注意图片尺寸要和训练阶段的一样, 并且预测的时候需要少量的good 图片用来计算score 的置信度


## 待修复bug列表
~~- 图片尺寸bug, 不支持长方形~~
~~- 无法在cpu上跑~~
- faiss 无法在gpu上跑， cpu跑的很慢

## 待优化功能
~~- 预测是把mask 合并到原图上~~

