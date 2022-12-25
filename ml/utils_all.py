import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
import sys 
sys.path.append('..')
from collections import OrderedDict
from sklearn.model_selection import train_test_split


def process_x(processing_type, x):
    """对x数据进行处理

    Args:
        processing_type (str): 处理类型或者说方法
        x (ndarray): 数据
    """
    if processing_type == "minmax":
        from sklearn.preprocessing import MinMaxScaler

        mm = MinMaxScaler()
        x = mm.fit_transform(x)
    elif processing_type == "stand":
        from sklearn.preprocessing import StandardScaler

        ss = StandardScaler()
        x = ss.fit_transform(x)
    elif processing_type == "maxabs":
        from sklearn.preprocessing import MaxAbsScaler

        ma = MaxAbsScaler()
        x = ma.fit_transform(x)
    else:
        x = x.apply(lambda x:x/eval(processing_type)).values #缩放
    return x


def get_x_y_data():
    """返回划分好的数据集和训练集

    Returns:
        _type_: _description_
    """
    x = np.load("data/x_data.npy")
    y = np.load("data/y_data.npy",allow_pickle=True)

    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.2, shuffle=True, random_state=20221224
    )
    return train_x, test_x, train_y, test_y


###开始训练与预测
def train_pre(model, train_x, test_x, train_y, test_y):
    """对模型进行训练,得到一些指标数据,这里追求全面

    Args:
        model (model): 模型
        train_x (array): 训练数据
        test_x (array): 测试数据
        train_y (array): 训练特征
        test_y (array): 测试特征
    """
    # 模型训练
    model.fit(train_x, train_y)

    print("---模型拟合表现---")
    print("模型最好的参数:{}".format(model.best_params_))
    print("模型最好的得分:{}".format(model.best_score_))
    print("模型最好的所有参数:{}".format(model.best_estimator_))
    print("最佳参数的索引:{}".format(model.best_index_))
    print("拟合的平均时间 (s):{}".format(round(model.cv_results_["mean_fit_time"].mean(), 3)))
    print(
        "预测的平均时间 (s):{}".format(round(model.cv_results_["mean_score_time"].mean(), 3))
    )

    # 获取最好的模型
    best_model = model.best_estimator_
    pred_train = best_model.predict(train_x)  # 获取标签
    accuracy1 = accuracy_score(train_y, pred_train)
    print("在训练集上的精确度: %.4f" % accuracy1)

    # 模型测试
    pred_test = best_model.predict(test_x)
    accuracy2 = accuracy_score(test_y, pred_test)
    print("在测试集上的精确度: %.4f" % accuracy2)
    try:
        prob = best_model.predict_prob(test_x)
    except AttributeError:
        print("没有pred_prob属性！！")
        Prob = None

    return (
        confusion_matrix(test_y, pred_test),
        (test_y, pred_test),
    )


###可视化,参考https://blog.csdn.net/Monk_donot_know/article/details/86614558
def classification_result(
    confusion_matrix, cm_type, test_result, model_name, result_path
):
    # 以下我自己知道实现的比较复杂，可以直接一个if，else解决，但是我觉得这样更加直观。
    for index, cm in enumerate(confusion_matrix):
        cm_info = OrderedDict()
        # cm才是混淆矩阵
        # 混淆矩阵
        print(cm_type[index] + "的混淆矩阵为:\n")
        print(cm)
        # print('cm的类型为:\n')
        # print(type(cm))
        # 注意json不能保存numpy的array结果
        # 这里采取转化为list保存
        cm_info[cm_type[index]] = cm.tolist()
        # TP:True Posirive:正确的肯定的分类数
        # TN:True Negatives:正确的否定的分类数
        # FP:False Positive:错误的肯定的分类数
        # FN:False Negatives:错误的否定
        TN = cm[0][0]
        FN = cm[1][0]
        TP = cm[1][1]
        FP = cm[0][1]
        # array类型取出具体的数值用item,我记得tensor也是
        cm_info["TN"] = TN.item()
        cm_info["FN"] = FN.item()
        cm_info["TP"] = TP.item()
        cm_info["FP"] = FP.item()
        # 二级指标

        # 准确率（Accuracy）(已完成)
        accuracy = accuracy_score(test_result[0], test_result[1])
        cm_info["accuracy"] = accuracy
        print("准确率为" + str(accuracy))

        # 精确率（Precision）——查准率（已完成）
        precision = precision_score(test_result[0], test_result[1])
        cm_info["precision"] = precision
        print("精确率为" + str(precision))

        # 查全率、召回率、反馈率（Recall）(已完成)
        recall = recall_score(test_result[0], test_result[1])
        cm_info["recall"] = recall
        print("召回率为" + str(recall))
        # 特异度（Specificity）(已完成)
        TNR = TN / (TN + FP)
        cm_info["TNR"] = TNR
        print("特异度为" + str(TNR))
        # FPR（假警报率）(已完成)
        FPR = FP / (FP + TN)
        cm_info["FPR"] = FPR
        print("假报警率为" + str(FPR))
        # TPR（真正率）(已完成)
        TPR = TP / (TP + FN)
        cm_info["TPR"] = TPR
        print("真正率为" + str(TPR))

        # 三级指标
        # F1_score(已完成)
        f1 = f1_score(test_result[0], test_result[1])
        cm_info["f1"] = f1
        print("f1分数为" + str(f1))
        # G-mean(在数据不平衡的时候,这个指标很有参考价值。)
        g_mean = np.sqrt(recall * TNR)
        cm_info["g_mean"] = g_mean
        print("g_mean值为" + str(g_mean))

        json_data = json.dumps(cm_info, indent=4)
        with open(
            os.path.join(result_path, model_name + "_test_res.json"),
            "w",
        ) as f:
            f.write(json_data)
        f.close
        print("成功保存测试结果！")


def plot_ROC(labels, preds, savepath):
    """
    Args:
        labels : ground truth
        preds : model prediction
        savepath : save path
    """
    # ROC曲线、Auc值(已完成)
    fpr, tpr, threshold = roc_curve(labels, preds)  ###计算真正率和假正率

    roc_auc = auc(fpr, tpr)  ###计算auc的值，auc就是曲线包围的面积，越大越好
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(
        fpr, tpr, color="darkorange", lw=lw, label="AUC = %0.2f" % roc_auc
    )  ###假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title("ROCs for Decision Tree")
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(savepath)  # 保存文件


if __name__ == '__main__':
    train_x, test_x, train_y, test_y = get_x_y_data()
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV
    dt = DecisionTreeClassifier()
    param_grid = {
        'max_depth': np.arange(29, 40),
        'min_samples_leaf': np.arange(1, 8), #1
        'min_samples_split': np.arange(2, 8) #3
    }
    #设置10折进行交叉验证
    model = GridSearchCV(dt, param_grid, cv=10, verbose=2, n_jobs=-1)
    train_pre(model,train_x, test_x, train_y, test_y)