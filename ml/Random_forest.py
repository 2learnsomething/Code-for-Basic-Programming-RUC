from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from utils_ml import tools
import numpy as np


###准备模型
def model_design():
    """得到设置交叉验证之后的模型

    Returns:
        model : 模型
    """
    rf = RandomForestClassifier(n_estimators=140,
                                criterion='gini',
                                max_depth=20,
                                min_samples_split=6,
                                min_samples_leaf=1,
                                random_state=1234,
                                n_jobs=-1)  #实例化
    #参数设置

    param_grid = {
        #'n_estimators': np.arange(120, 160, 10),
        #'criterion': ['gini','entropy']
        #'max_depth': np.arange(17, 30, 3),
        #'min_samples_split': np.arange(2, 16, 2),
        #'min_samples_leaf': np.arange(1, 15, 2),
        'max_features': np.arange(0.1, 1.1, 0.3)
    }

    #设置10折进行交叉验证
    model = GridSearchCV(rf, param_grid, cv=10, verbose=2, n_jobs=-1)
    # 进行预测
    return model


def main():
    tool = tools()
    res_path = 'res\confuse_matrix'
    model_name = 'RF'
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
