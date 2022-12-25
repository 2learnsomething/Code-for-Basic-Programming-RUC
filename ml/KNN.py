from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from utils_ml import tools


###准备模型
def model_design():
    """得到设置交叉验证之后的模型

    Returns:
        model : 模型
    """
    knn = KNeighborsClassifier(n_jobs=-1)  #实例化
    #参数设置

    param_grid = [
        {  # 需遍历10次
            'weights': ['uniform'],  # 参数取值范围
            'n_neighbors': [i for i in range(1, 11)]  # 使用其他方式如np.arange()也可以
            # 这里没有p参数
        },
        {  # 需遍历50次
            'weights': ['distance'],
            'n_neighbors': [i for i in range(1, 11)],
            'p': [i for i in range(1, 6)]
        }
    ]

    #设置10折进行交叉验证
    model = GridSearchCV(knn, param_grid, cv=10, verbose=2, n_jobs=-1)
    # 进行预测
    return model


def main():
    tool = tools()
    res_path = 'res\confuse_matrix'
    model_name = 'KNN'
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
