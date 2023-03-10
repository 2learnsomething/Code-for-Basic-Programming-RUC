import matplotlib.pyplot as plt
import os
import json


def get_result(train_or_test='test', data_type='precision'):
    """获取想要可视化的分类结果数据

    Args:
        train_or_test (str, optional): 可视化测试集结果还是训练集结果. Defaults to 'test'.
        data_type (str, optional): 想要可视化的数据. Defaults to 'precision'.

    Returns:
        tuple: 模型名称和数据
    """
    pass


def name_tran(data_type):
    """将指标类型转化为中文

    Args:
        data_type (str): 需要转化的指标类型

    Returns:
        str: 中文字符串
    """
    if data_type == 'precision':
        return '精确率'
    elif data_type == 'accuracy':
        return '准确率'
    elif data_type == 'recall':
        return '召回率'
    elif data_type == 'f1':
        return 'F1得分'
    elif data_type == 'TNR':
        return '特异度'
    elif data_type == 'FPR':
        return '假报警率'


def vis_data(model_name, data_list, data_type):
    """对分类指标进行绘制可视化

    Args:
        model_name (list): 模型列表
        data_list (list): 可视化的数据
        data_type (str): 可视化的指标
    """
    plt.figure()
    plt.rcParams["font.sans-serif"] = ["SimHei"]  #设置字体
    plt.rcParams["axes.unicode_minus"] = False  #该语句解决图像中的“-”负号的乱码问题
    plt.xlabel('模型类型')
    data_type = name_tran(data_type)
    plt.ylabel(data_type)
    plt.plot(model_name, data_list, 'r*-.')
    for name, data in zip(model_name, data_list):
        plt.text(name, data, round(data, 2), color='blue', fontsize=8)
    #plt.show()
    plt.savefig(os.path.join('res\fig', data_type + '.pdf'))
    print('保存图片成功')


def main():
    train_or_test = 'test'
    data_type_list = ['precision', 'accuracy', 'recall', 'f1']


if __name__ == '__main__':
    main()