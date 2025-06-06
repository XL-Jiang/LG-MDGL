import torch
import torchmetrics
from torchmetrics.classification import MulticlassSpecificity,roc
from torchmetrics.functional.classification.roc import binary_roc,multiclass_roc
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import itertools
from torchmetrics import ROC
from sklearn.metrics import precision_recall_fscore_support,roc_curve, roc_auc_score,auc

def torchmetrics_accuracy(preds, labels):
    acc = torchmetrics.functional.accuracy(preds, labels,task = 'binary')
    return acc

def torchmetrics_spef(preds, labels):
    metric = MulticlassSpecificity(num_classes=2).cuda()
    spef = metric(preds, labels)
    return spef

def torchmetrics_auc(preds, labels):
    auc = torchmetrics.functional.auroc(preds, labels, task="multiclass", num_classes=2)
    return auc

def confusion_matrix(preds, labels):
    conf_matrix = torch.zeros(2, 2)
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[t, p] += 1 
    return conf_matrix
def plot_confusion_matrix(cm, normalize=False, title='Confusion matrix', cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : computer the value of confusion matrix
    - normalize : True: %, False: 123
    """
    classes = ['0:ASD','1:TC']
    if normalize:
        cm = cm.numpy()
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else '.0f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def correct_num(preds, labels):
    #numpy
    """Accuracy, auc with masking.Acc of the masked samples"""
    #correct_prediction = np.equal(np.argmax(preds, 1), labels).astype(np.float32)
    correct_prediction = np.equal(preds, labels).astype(np.float32)
    return np.sum(correct_prediction)

def prf(preds, labels, is_logit=True):
    '''

    :param preds: 预测标签，0,1,0,1，...
    :param labels: 真实标签
    :param is_logit:
    :return:
    '''
    ''' input: logits, labels  ''' 
    #pred_lab= np.argmax(preds, 1)，将预测概率大的列作为标签
    p,r,f,s  = precision_recall_fscore_support(labels, preds, average='binary',zero_division=0)
    return [p,r,f]
#zero_division='warn' / 0 控制警告行为,提示模型在某些类别上预测样本为0


# def plot_embedding(data, label, title):
#     # 绘制节点分布散点图
#     plt.figure(figsize=(3.46, 2.59))  # 设置图形的长宽为8.8cm*6.59cm
#     plt.rcParams['font.family'] = 'Times New Roman'
#     plt.rcParams['font.size'] = 8  # 设置全局字体大小为12
#     p = []
#     p2 = []
#     # p = [[0] for _ in range(10)]
#     # p2 = [[0] for _ in range(10)]
#     for i in range(len(label)):
#         if label[i] == 0:  # 如果标签为0，该点坐标为黄色
#             p.append(plt.scatter(data[i, 0], data[i, 1], lw=0.1, c='#0071C5'))#FFD700
#         # p = plt.scatter(data_normalized[i, 0], data_normalized[i, 1], lw=0.1, c='#FFD700')  #scatter返回一个散列点对象，代表绘制的散点图 , alpha=0.8
#         # data_normalized[i, 0], data_normalized[i, 1] ,代表x轴与y轴坐标
#         elif label[i] == 1:
#             p2.append(plt.scatter(data[i, 0], data[i, 1], lw=0.1, c='#DB4437'))
#             # p2 = plt.scatter(data_normalized[i, 0], data_normalized[i, 1], lw=0.1, c='#800080')
#     plt.legend((p[0], p2[0]), ('ASD', 'HC'))
#     # 移除刻度
#     plt.xticks([])
#     plt.yticks([])
#     plt.savefig('./figures/ASD/{:s}.tiff'.format(title),dpi=300)

def plot_embedding1(data, label, title):
    # 绘制节点分布散点图
    # 绘制节点分布散点图
    plt.figure(figsize=(3.46, 2.59))  # 设置图形的长宽为8.8cm*6.59cm
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 8  # 设置全局字体大小为12
    p = []
    p2 = []
    # p = [[0] for _ in range(10)]
    # p2 = [[0] for _ in range(10)]
    for i in range(len(label)):
        if label[i] == 0:  # 如果标签为0，该点坐标为黄色
            p.append(plt.scatter(data[i, 0], data[i, 1], lw=0.1, c='#0071C5'))
        # p = plt.scatter(data_normalized[i, 0], data_normalized[i, 1], lw=0.1, c='#FFD700')  #scatter返回一个散列点对象，代表绘制的散点图 , alpha=0.8
        # data_normalized[i, 0], data_normalized[i, 1] ,代表x轴与y轴坐标
        elif label[i] == 1:
            p2.append(plt.scatter(data[i, 0], data[i, 1], lw=0.1, c='#DB4437'))
            # p2 = plt.scatter(data_normalized[i, 0], data_normalized[i, 1], lw=0.1, c='#800080')
    plt.legend((p[0], p2[0]), ('HC', 'ASD'))
    # 移除刻度
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./figures/ASD/{:s}.tiff'.format(title),dpi=300)
def plot_embedding(data, label, title):
    plt.figure()
    plt.rcParams['font.family'] = 'Times New Roman'
    p = []
    p2 = []
    for i in range(len(label)):
        if label[i] == 1:
            p.append(plt.scatter(data[i, 0], data[i, 1], lw=0.1, c='#0071C5'))
        elif label[i] == 0:
            p2.append(plt.scatter(data[i, 0], data[i, 1], lw=0.1, c='#DB4437'))
    plt.legend((p[0], p2[0]), ('ASD', 'HC'))
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./figures/{:s}.tiff'.format(title),dpi=100)

def plot_roc_curve(y_true, y_score,auc):
    #绘制ROC曲线
    fpr, tpr, thresholds = roc.binary_roc(y_true, y_score)
    # fpr, tpr, thresholds = roc_curve(y_true, y_score)  # 计算ROC曲线
    # auc = torchmetrics_auc(y_score, y_true)
    # auc = roc_auc_score(y_true, y_score)  # 计算AUC值
    plt.plot(fpr, tpr, marker='.', label='ROC curve (area = %0.2f)' % auc)  # 绘制ROC曲线
    plt.plot([0, 1], [0, 1], linestyle='--')  # 绘制随机猜测的曲线（对角线）
    plt.legend(loc="lower right")  # 添加图例
    plt.title('Receiver Operating Characteristic')  # 添加标题
    plt.xlabel('False Positive Rate')  # 添加x轴标签
    plt.ylabel('True Positive Rate')  # 添加y轴标签
    plt.grid(True)  # 添加网格线
    plt.show()  # 显示图形
    return auc

def plot_ROC(labels_list, logits_list,auc_list):
    plt.figure()
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12  # 设置字体大小为14
    plt.rcParams['font.weight'] = 'bold'  # 设置字体加粗
    color_map = cm.get_cmap('tab10')
    for i in range(len(labels_list)):
        fpr, tpr, _ = roc_curve(labels_list[i], logits_list[i])
        roc_auc = auc(fpr, tpr)
        color = color_map(i)
        plt.plot(fpr, tpr,color = color, label='ROC(AUC = %0.2f)fold %d' % (roc_auc, i+1))
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='AUC=0.5')
    plt.xlabel('False Positive Rate', fontsize=12,fontweight='bold')#, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12,fontweight='bold')#, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=12)  # 加大坐标轴上数字标签字体
    plt.legend(loc="lower right")
    plt.savefig('./figures1/ROC.tiff', dpi=600)
    plt.savefig('./figures1/ROC.eps', format='eps', dpi=10000)
    plt.show()

def plot_ROC1(labels_list, logits_list, auc_list):
    plt.figure(dpi=1200)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 5
    color_map = cm.get_cmap('tab10')
    for i in range(len(labels_list)):
        fpr, tpr, _ = roc_curve(labels_list[i], logits_list[i])
        roc_auc = auc(fpr, tpr)
        color = color_map(i)
        plt.plot(fpr, tpr, color=color, label='ROC (AUC = %0.2f) fold %d' % (roc_auc, i+1))
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='AUC = 0.5')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig('./figures/ASD/ROC.tiff', format='tiff')
    plt.show()

def plot_mean_roc_multi_rounds(y_true_list, y_score_list):
    mean_fpr = np.linspace(0, 1, 100)
    tpr_list = []
    for y_true, y_scores in zip(y_true_list, y_score_list):
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        tpr_list.append(np.interp(mean_fpr, fpr, tpr))

    mean_tpr = np.mean(tpr_list, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    plt.figure()
    plt.plot(mean_fpr, mean_tpr, color='b', label='Mean ROC (AUC = %0.2f)' % mean_auc)
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Mean Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()

def plot_roc(predictions, targets):
    roc = ROC(compute_on_step=False)
    roc.update(predictions, targets)
    roc_curve = roc.compute()
    fpr, tpr, _ = roc_curve
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

