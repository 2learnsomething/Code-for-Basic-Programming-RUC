import os
import sys

sys.path.append('.')
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from utils_bayes import train_pre, classification_result, plot_ROC


#获取数据
def get_x_y_data():
    """返回划分好的训练集和测试集,目前不考虑验证集(validation set)

    Args:
        company_code (list): _description_
        path (str): _description_
        columns (str): _description_
        processing_type (str): 预处理的类型

    Returns:
        ndarray: 划分好的训练集和测试集的元组
    """
    #这个地方后续需要修改，先改整体逻辑
    x = np.load('/new_python_for_gnn/毕设code/technical_analysis/x_data.npy')
    y = np.load('/new_python_for_gnn/毕设code/technical_analysis/y_data.npy')

    train_x, test_x, train_y, test_y = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=1234)
    return train_x, test_x, train_y, test_y


###准备模型
def model_design():
    """得到设置交叉验证之后的模型

    Returns:
        model : 模型
    """
    parameters = {'binarize': [0.001, 0.01, 0.1, 1]}
    bn = BernoulliNB() #实例化
    model = GridSearchCV(bn, parameters, refit = True, cv = 10, verbose = 1, n_jobs = -1)
    # 进行预测
    return model


def main():
    cm_type = ['train_result', 'test_result']
    figure_path = 'res\fig'
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
    classification_result(confusion_matrix, cm_type, train_result, test_result,
                          'Bayes')
    #roc曲线可视化
    print('可视化ing...')
    for index, lable in enumerate([train_result, test_result]):
        if index == 0 and train_result[-1]:
            plot_ROC(labels=lable[0],
                     preds=lable[2],
                     savepath=os.path.join(figure_path, 'Bayes_train_roc.jpg'))
        elif index == 1 and test_result[-1]:
            plot_ROC(labels=lable[0],
                     preds=lable[2],
                     savepath=os.path.join(figure_path, 'Bayes_test_roc.jpg'))
        else:
            print('模型没法预测分类概率，没法进行可视化')
            continue
    print('全部结束！！！！')


if __name__ == '__main__':
    main()
