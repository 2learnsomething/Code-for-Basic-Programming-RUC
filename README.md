# Code-for-Basic-Programming-RUC
本项目为ruc大数据研究院编程基础的课程大作业项目，做的主要是干豆的分类实验，数据来源https://archive.ics.uci.edu/ml/datasets/dry+bean+dataset，数据总共有1w多条数据，包含16个特征，7个类型干豆
更多的数据方面的说明可以见Dry_Bean_Dataset.txt，另外数据文件见Dry_Bean_Dataset.xlsx
## 数据说明
原始数据为Dry_Bean_Dataset.xlsx，可以打开查看
另外的data文件夹，则是将数据存储为npy格式方便模型训练和测试读取
## 环境说明
本文是新建了一个虚拟环境，python版本为3.9.15,包就是常用的numpy，pandas，matplotlib，seaborn以及sklearn
## ml文件夹
本文件夹主要包含了我实现的各种机器学习算法模型，直接运行即可，但是主要可能的路径错误
## res文件夹
主要包含两个文件夹，一个保存图片相关的结果，另外一个是诸如accuracy等指标数据的json文件，可以用相关编辑器打开查看。
## 额外说明
res_vis.py则是要对accuracy等指标进行可视化绘图，res_vis.ipynb是相应的jupyter版本。
process.ipynb 主要是数据的探索性分析