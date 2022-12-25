from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from utils_ml import tools


###准备模型
def model_design():
    """得到设置交叉验证之后的模型

    Returns:
        model : 模型
    """
    gbdt = GradientBoostingClassifier(n_estimators=150,
                                      max_depth=8,
                                      min_samples_split=80)  #实例化
    #参数设置
    param_grid = {
        #'n_estimators':range(130,171,10), #170
        #'max_depth':range(2,13,2), # 12
        #'min_samples_split':range(10,191,20),
        'min_samples_leaf': range(10, 101, 10),  #60
        'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]  #忘记记了，假设0.85
    }

    #设置10折进行交叉验证
    model = GridSearchCV(gbdt, param_grid, cv=10, verbose=2, n_jobs=-1)
    # 进行预测
    return model


def main():
    tool = tools()
    res_path = 'res\confuse_matrix'
    model_name = 'GBDT'
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
