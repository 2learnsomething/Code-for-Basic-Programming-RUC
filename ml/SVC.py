import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from utils_ml import tools


###准备模型
def model_design():
    """得到设置交叉验证之后的模型

    Returns:
        model : 模型
    """
    svc = SVC(probability=True)  #实例化
    #参数设置

    param_grid = [
        {
            'kernel': ['rbf'],
            'gamma': [1, 1.5, 2, 2.5, 3],
            'C': [200]  # rbf,gamma=3的时候最好
        },
        {
            'kernel': ['linear'],
            'C': [300]
        },
        {
            'kernel': ['poly'],
            'degree': [1],
            'gamma': [1, 1.5, 2, 2.5, 3]
        }
    ]

    #设置10折进行交叉验证
    model = GridSearchCV(svc, param_grid, cv=10, verbose=2, n_jobs=-1)
    # 进行预测
    return model


def main():
    tool = tools()
    res_path = 'res\confuse_matrix'
    model_name = 'SVC'
    #获取数据集
    print('获取数据ing...')
    train_x, test_x, train_y, test_y = tool.get_x_y_data()
    #获取模型
    print('获取模型ing...')
    model = model_design()
    #开始训练
    print('开始训练ing...')
    test_result = tool.train_pre(model, train_x, test_x, train_y, test_y)
    #保存模型
    print('保存模型ing...')
    tool.classification_result(test_result, model_name, res_path)
    print('全部结束！！！！')


if __name__ == '__main__':
    main()