import os
import sys 
sys.path.append('.')
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from utils_ml import classification_result
np.random.seed(1234)

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
    #由于第一个实现的是decision——tree,已经保存了相关的数据文件，所以这里直接读取，就可以节省数据concanate的时间
    x = np.load('technical_analysis/x_data.npy')
    y = np.load('technical_analysis/y_data.npy')

    train_x, test_x, train_y, test_y = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=1234)
    return train_x, test_x, train_y, test_y

def main():
    cm_type = ['train_result', 'test_result']
    #确定公司代码
    #print('确定公司ing...')
    #company_list = new_left_company(company_final)
    #company_list = list(map(lambda x: x.replace('.csv', ''), company_list))
    #company_use = choice(company_list)
    #获取数据集
    print('获取数据ing...')
    train_x, test_x, train_y, test_y = get_x_y_data()
    y_train_guess = np.random.randint(2,size=train_y.size)
    y_test_guess = np.random.randint(2,size=test_y.size)
    confusion_matrix_train = confusion_matrix(train_y,y_train_guess)
    confusion_matrix_test = confusion_matrix(test_y,y_test_guess)
    confusion_matrix_all = (confusion_matrix_train,confusion_matrix_test)
    train_result = (train_y,y_train_guess,np.array([1,2,3]),False)
    test_result = (test_y,y_test_guess,np.array([1,2,3]),False)
    classification_result(confusion_matrix_all,cm_type,train_result,test_result,'Guess')



if __name__ == '__main__':
    main()
