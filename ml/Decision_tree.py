import pandas as pd
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import sys

sys.path.append(".")
from utils_all import train_pre,get_x_y_data

###准备模型
def model_design():
    """得到设置交叉验证之后的模型

    Returns:
        model : 模型
    """
    decisiontree = DecisionTreeClassifier(min_samples_leaf=1,min_samples_split=3,random_state=1234)  #实例化
    #参数设置
    param_grid = {
        'max_depth': np.arange(29, 40),
        'min_samples_leaf': np.arange(1, 8), #1
        'min_samples_split': np.arange(2, 8) #3
    }
    #设置10折进行交叉验证
    model = GridSearchCV(decisiontree, param_grid, cv=10, verbose=2, n_jobs=-1)
    # 进行预测
    return model


def main():
    cm_type = 'test_result'
    model_name = 'Decision_tree'
    #获取数据集
    print('获取数据ing...')
    train_x, test_x, train_y, test_y = get_x_y_data()
    #获取模型
    print('获取模型ing...')
    model = model_design()
    #开始训练
    print('开始训练ing...')
    confusion_matrix, train_result, test_result = train_pre(
        model, train_x, test_x, train_y, test_y)
    #保存模型
    print('保存模型ing...')

    #roc曲线可视化
    print('可视化ing...')
    
    print('全部结束！！！！')


if __name__ == '__main__':
    main()