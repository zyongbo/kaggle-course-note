# Kaggle竞赛案例深度剖析
> 轻松赢得让大厂面试官双眼放光的竞赛经验

课程主页：https://coding.imooc.com/learn/list/504.html

Kaggle官网：https://www.kaggle.com/

## 课程代码

- [2-5：XGBoost 、LightGBM和Catboost](https://git.imooc.com/coding-504/kaggle-course-note/src/master/code/2-quick-study-lgbm-xgb-and-catboost-lb-1-66.ipynb), [在线运行版本](https://www.kaggle.com/julian3833/2-quick-study-lgbm-xgb-and-catboost-lb-1-66)
- [2-6：数据划分与交叉验证](https://git.imooc.com/coding-504/kaggle-course-note/src/master/code/kaggle-ch3.ipynb), [在线运行版本](https://www.kaggle.com/finlay/kaggle-ch3/)
- [2-7：模型集成方法](https://git.imooc.com/coding-504/kaggle-course-note/src/master/code/stacking_starter.py), [在线运行版本](https://www.kaggle.com/abhilashawasthi/stacking-starter)
- [2-8：比赛案例Titanic幸存乘客预测](https://git.imooc.com/coding-504/kaggle-course-note/src/master/code/titanic-starter.ipynb), [在线运行版本](https://www.kaggle.com/finlay/titanic-starter/)
- [3-1/3-2/3-3：结构化技能之特征工程/特征筛选](https://git.imooc.com/coding-504/kaggle-course-note/src/master/code/feature-engineer-starter.ipynb)
- [3-4 Instant Gratification赛题介绍](https://git.imooc.com/coding-504/kaggle-course-note/src/master/code/instant-gratification-eda.ipynb)
- [3-5 Instant Gratification赛题实践](https://git.imooc.com/coding-504/kaggle-course-note/src/master/code/instant-gratification-qda-starter.ipynb)
- [3-8 IEEE-CIS Fraud Detection赛题介绍](https://git.imooc.com/coding-504/kaggle-course-note/src/master/code/ieee-fraud-detection-eda.ipynb)
- [3-9 IEEE-CIS Fraud Detection赛题实践](https://git.imooc.com/coding-504/kaggle-course-note/src/master/code/ieee-fraud-detection-starter.ipynb)
- [4-3 Quora Question Pairs赛题介绍](https://git.imooc.com/coding-504/kaggle-course-note/src/master/code/quora-pairs-eda.ipynb)
- [4-4 Quora Question Pairs赛题实践](https://git.imooc.com/coding-504/kaggle-course-note/src/master/code/quora-pairs-starter.ipynb)
- [4-7 Quora Insincere Questions Classification赛题介绍](https://git.imooc.com/coding-504/kaggle-course-note/src/master/code/simple-exploration-notebook-qiqc.ipynb), [在线运行版本](https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-qiqc)
- [4-8 Quora Insincere Questions Classification赛题实践](https://git.imooc.com/coding-504/kaggle-course-note/src/master/code/qiqc-text-modelling-in-pytorch.ipynb), [在线版本](https://www.kaggle.com/finlay/qiqc-text-modelling-in-pytorch/)
- [5-2/5-3 语音特征处理](https://git.imooc.com/coding-504/kaggle-course-note/src/master/code/audio-basic.ipynb)
- [5-4 语音模型和数据扩增方法](https://git.imooc.com/coding-504/kaggle-course-note/src/master/code/audio-aug.ipynb)
- 5-5 Cornell Birdcall Identification赛题介绍, [在线运行版本](https://www.kaggle.com/rohitsingh9990/eda-visualizations-simple-baseline)
- 5-6 Cornell Birdcall Identification动手实践, [在线运行版本](https://www.kaggle.com/hidehisaarai1213/introduction-to-sound-event-detection)


## 课程数据集


- Titanic数据集链接链接: https://pan.baidu.com/s/1esdu5AQNa3CIbNpVXtbNAw  密码: 2gje
- Instant Gratification数据集链接: https://pan.baidu.com/s/1MfC2CikWeKMnjJAk2jD5oQ  密码: 1e4t
- IEEE-CIS Fraud Detection数据集链接：链接: https://pan.baidu.com/s/1RduL9UeNaILSwd4F2YEUFQ  密码: w6gc
- Quora Question Pairs数据集链接: https://pan.baidu.com/s/1RPr58Od9nxmwfjCunusFdA  密码: 872b


## 库使用

### timm库：图像分类


列举模型：
```
import timm
from pprint import pprint
model_names = timm.list_models('*eff*t*')
pprint(model_names)
```

加载模型：
```
import timm
m = timm.create_model('resnet18', pretrained=True)
m.eval()
```

### smp库：语义分割

https://github.com/qubvel/segmentation_models.pytorch

```
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=3,                      # model output channels (number of classes in your dataset)
)
```