import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
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

class tools():
    def __init__(self) -> None:
        pass
    
    def process_x(self,processing_type, x):
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


    def get_x_y_data(self):
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
    def train_pre(self,model, train_x, test_x, train_y, test_y):
        """对模型进行训练,得到一些指标数据,这里追求全面

        Args:
            model (model): CV模型
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

        return (test_y, pred_test)


    ###可视化,参考https://blog.csdn.net/Monk_donot_know/article/details/86614558
    def classification_result(self,test_result, model_name, result_path):
        true_lable,prediction = test_result
        #存储容器
        cm_info = OrderedDict()
        #一级指标
        confuse_matrix = confusion_matrix(true_lable, prediction)
        cm_info['confuse_matrix'] = confuse_matrix.tolist()
        # 二级指标
        accuracy = accuracy_score(true_lable, prediction)
        # 准确率（Accuracy）(已完成)
        cm_info["accuracy"] = accuracy
        print("准确率为" + str(accuracy))
        
        measure_result = classification_report(true_lable, prediction)
        print('measure_result = \n', measure_result)

        print("----------------------------- precision(精确率)-----------------------------")
        precision_score_average_None = precision_score(true_lable, prediction,average=None)
        precision_score_average_micro = precision_score(true_lable, prediction, average='micro')
        precision_score_average_macro = precision_score(true_lable, prediction, average='macro')
        precision_score_average_weighted = precision_score(true_lable, prediction, average='weighted')
        cm_info['precision_score_average_None'] = precision_score_average_None.tolist()
        cm_info['precision_score_average_micro'] = precision_score_average_micro
        cm_info['precision_score_average_macro'] = precision_score_average_macro
        cm_info['precision_score_average_weighted'] = precision_score_average_weighted
        print('precision_score_average_None = ', precision_score_average_None)
        print('precision_score_average_micro = ', precision_score_average_micro)
        print('precision_score_average_macro = ', precision_score_average_macro)
        print('precision_score_average_weighted = ', precision_score_average_weighted)

        print("\n\n----------------------------- recall(召回率)-----------------------------")
        recall_score_average_None = recall_score(true_lable, prediction,average=None)
        recall_score_average_micro = recall_score(true_lable, prediction, average='micro')
        recall_score_average_macro = recall_score(true_lable, prediction, average='macro')
        recall_score_average_weighted = recall_score(true_lable, prediction, average='weighted')
        cm_info['recall_score_average_None'] = recall_score_average_None.tolist()
        cm_info['recall_score_average_micro'] =recall_score_average_micro
        cm_info['recall_score_average_macro'] =recall_score_average_macro
        cm_info['recall_score_average_weighted'] = recall_score_average_weighted
        print('recall_score_average_None = ', recall_score_average_None)
        print('recall_score_average_micro = ', recall_score_average_micro)
        print('recall_score_average_macro = ', recall_score_average_macro)
        print('recall_score_average_weighted = ', recall_score_average_weighted)

        print("\n\n----------------------------- F1-value-----------------------------")
        f1_score_average_None = f1_score(true_lable, prediction,average=None)
        f1_score_average_micro = f1_score(true_lable, prediction, average='micro')
        f1_score_average_macro = f1_score(true_lable, prediction, average='macro')
        f1_score_average_weighted = f1_score(true_lable, prediction, average='weighted')
        cm_info['f1_score_average_None'] = f1_score_average_None.tolist()
        cm_info['f1_score_average_micro'] =f1_score_average_micro
        cm_info['f1_score_average_macro'] =f1_score_average_macro
        cm_info['f1_score_average_weighted'] =f1_score_average_weighted
        print('f1_score_average_None = ', f1_score_average_None)
        print('f1_score_average_micro = ', f1_score_average_micro)
        print('f1_score_average_macro = ', f1_score_average_macro)
        print('f1_score_average_weighted = ', f1_score_average_weighted)

        json_data = json.dumps(cm_info, indent=4)
        with open(
            os.path.join(result_path, model_name + ".json"),
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
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
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
        'max_depth': np.arange(30, 32),
        'min_samples_leaf': np.arange(6, 8), 
        'min_samples_split': np.arange(5, 7) 
    }
    #设置10折进行交叉验证
    model = GridSearchCV(dt, param_grid, cv=10, verbose=2, n_jobs=-1)
    test_res = train_pre(model,train_x, test_x, train_y, test_y)
    classification_result(test_result=test_res,model_name='decison_tree',result_path='res\confuse_matrix')
    plot_ROC(test_res[0],test_res[1],'res\fig')