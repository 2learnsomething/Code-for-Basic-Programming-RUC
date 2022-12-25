import numpy as np
from sklearn.metrics import confusion_matrix
from utils_ml import tools

np.random.seed(1234)


def main():
    tool = tools()
    res_path = 'res\confuse_matrix'
    model_name = 'Guess'
    print('获取数据ing...')
    train_x, test_x, train_y, test_y = tool.get_x_y_data()
    n_classes = 7
    y_test_guess = np.random.choice(np.unique(train_y), size=test_y.size)
    test_result = (test_y, y_test_guess)
    #保存模型
    print('保存模型ing...')
    tool.classification_result(test_result, model_name, res_path)
    print('全部结束！！！！')


if __name__ == '__main__':
    main()
