from math import gamma
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from utils_ml import tools


###准备模型
def model_design():
    """得到设置交叉验证之后的模型

    Returns:
        model : 模型
    """
    lr = LogisticRegression(solver='liblinear')  #实例化

    #参数的搜索范围
    penaltys = ['l1', 'l2']
    Cs = [0.1, 1, 10, 100, 1000]
    #调优的参数集合，搜索网格为x5，在网格上的交叉点进行搜索
    param_grid = dict(penalty=penaltys, C=Cs)

    #设置10折进行交叉验证
    model = GridSearchCV(lr,
                         param_grid,
                         cv=10,
                         scoring='neg_log_loss',
                         verbose=2,
                         n_jobs=-1)
    # 进行预测
    return model


def main():
    tool = tools()
    res_path = 'res\confuse_matrix'
    model_name = 'LR'
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
