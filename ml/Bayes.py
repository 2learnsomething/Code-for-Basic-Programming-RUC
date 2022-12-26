import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV
from utils_ml import tools
import warnings
warnings.filterwarnings('ignore')

###准备模型
def model_design():
    """得到设置交叉验证之后的模型

    Returns:
        model : 模型
    """
    parameters = {'binarize': np.linspace(0.1, 0.9, 9)}
    bn = BernoulliNB()  #实例化
    model = GridSearchCV(bn,
                         parameters,
                         refit=True,
                         cv=10,
                         verbose=1,
                         n_jobs=-1)
    # 进行预测
    return model


def main():
    tool = tools()
    model_name = 'Bayes'
    res_path = 'res\confuse_matrix'
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
