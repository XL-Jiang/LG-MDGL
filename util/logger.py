import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
def draw_loss(train_loss_array):
    """
     为每个样本单独绘制一张图，展示其在所有轮次的损失值。

     参数:
     train_loss_array (numpy.ndarray): 一个形状为(num_samples, num_epochs)的数组，
                                      其中num_samples是样本数，num_epochs是轮次数。
     """
    # 获取样本数（即数组的行数）
    num_samples = train_loss_array.shape[0]
    # 遍历每个样本
    for i in range(num_samples):
        # 绘制当前样本的损失值曲线
        plt.figure(figsize=(10, 5))  # 创建一个新的图形
        plt.plot(train_loss_array[i], label=f'Sample {i + 1}')
        plt.title(f'Loss Curve for Sample {i + 1}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()  # 显示当前样本的图形



class Logger(object):
    #日志记录器，用于收集和评估模型的性能指标
    def __init__(self, k_fold=None, num_classes=None):
        super().__init__()
        self.k_fold = k_fold
        self.num_classes = num_classes
        self.initialize(k=None)


    def __call__(self, **kwargs):
        if len(kwargs)==0:
            self.get()
        else:
            self.add(**kwargs)


    def _initialize_metric_dict(self):
        #返回一个包含空列表的字典，用于存储预测结果、真实标签和预测概率
        return {'pred':[], 'true':[], 'prob':[]}#,'binary_true':[]}


    def initialize(self, k=None):
        if self.k_fold is None:
            #是否进行交叉验证（k_fold）初始化日志记录器的状态
            self.samples = self._initialize_metric_dict()
        else:
            if k is None:
                #如果不进行交叉验证，初始化一个字典来存储所有指标
                self.samples = {}
                for _k in range(self.k_fold):
                    self.samples[_k] = self._initialize_metric_dict()
            else:
                #如果进行交叉验证，为每个折初始化一个字典
                self.samples[k] = self._initialize_metric_dict()


    def add(self, k=None, **kwargs):
        #向日志记录器添加数据
        if self.k_fold is None:
            #如果 k_fold 为 None，则直接将数据添加到 samples 字典中
            for sample, value in kwargs.items():
                self.samples[sample].append(value)
        else:
            #如果进行了交叉验证，则将数据添加到指定折的字典中
            assert k in list(range(self.k_fold))
            for sample, value in kwargs.items():
                self.samples[k][sample].append(value)

    def specificity(self,y_true, y_pred):
        # 检查 y_true 和 y_pred 是否为字典
        if isinstance(y_true, dict) and isinstance(y_pred, dict):
            # 如果是字典，将字典中的数组合并成一个整体的数组
            y_true_array = np.concatenate(list(y_true.values()))
            y_pred_array = np.concatenate(list(y_pred.values()))
        else:
            # 如果不是字典，直接使用 y_true 和 y_pred
            y_true_array = y_true
            y_pred_array = y_pred

        cm = confusion_matrix(y_true_array, y_pred_array)
        tn, fp, fn, tp = cm.ravel()
        specificity_score = tn / (tn + fp)
        sensitivity_score = tp / (tp + fn)
        return specificity_score

    def sensitivity(self,y_true, y_pred):
        # 检查 y_true 和 y_pred 是否为字典
        if isinstance(y_true, dict) and isinstance(y_pred, dict):
            # 如果是字典，将字典中的数组合并成一个整体的数组
            y_true_array = np.concatenate(list(y_true.values()))
            y_pred_array = np.concatenate(list(y_pred.values()))
        else:
            # 如果不是字典，直接使用 y_true 和 y_pred
            y_true_array = y_true
            y_pred_array = y_pred

        cm = confusion_matrix(y_true_array, y_pred_array)
        tn, fp, fn, tp = cm.ravel()
        specificity_score = tn / (tn + fp)
        sensitivity_score = tp / (tp + fn)
        return sensitivity_score

    def get(self, k=None, initialize=False):
        #获取存储在日志记录器中的所有数据
        if self.k_fold is None:
            #如果不进行交叉验证，返回所有数据
            true = np.concatenate(self.samples['true'])
            pred = np.concatenate(self.samples['pred'])
            prob = np.concatenate(self.samples['prob'])
            # binary_true = np.concatenate(self.samples['binary_true'])
        else:
            if k is None:
                #如果进行了交叉验证，未指定某一折，则可以返回所有折的数据
                true, pred, prob = {}, {}, {}
                for k in range(self.k_fold):
                    true[k] = np.concatenate(self.samples[k]['true'])
                    pred[k] = np.concatenate(self.samples[k]['pred'])
                    prob[k] = np.concatenate(self.samples[k]['prob'])
                    # binary_true = np.concatenate(self.samples[k]['binary_true'])
            else:
                #如果进行了交叉验证，并指定某一折，则可以返回该折的数据
                true = np.concatenate(self.samples[k]['true'])
                pred = np.concatenate(self.samples[k]['pred'])
                prob = np.concatenate(self.samples[k]['prob'])
                # binary_true = np.concatenate(self.samples[k]['binary_true'])

        if initialize:
            self.initialize(k)

        return dict(true=true, pred=pred, prob=prob)#,binary_true=binary_true)


    def evaluate(self, k=None, initialize=False, option='mean'):
        samples = self.get(k)
        if not self.k_fold is None and k is None:
            #如果不进行交叉验证，直接计算指标
            if option=='mean': aggregate = np.mean
            elif option=='std': aggregate = np.std
            else: raise
            
            accuracy = aggregate([metrics.accuracy_score(samples['true'][k], samples['pred'][k]) for k in range(self.k_fold)])
            precision = aggregate([metrics.precision_score(samples['true'][k], samples['pred'][k], average='binary' if self.num_classes==2 else 'micro') for k in range(self.k_fold)])
            recall = aggregate([metrics.recall_score(samples['true'][k], samples['pred'][k],pos_label=1, average='binary' if self.num_classes==2 else 'micro') for k in range(self.k_fold)])
            roc_auc = aggregate([metrics.roc_auc_score(samples['true'][k], samples['prob'][k][:,1]) for k in range(self.k_fold)]) if self.num_classes==2 else np.mean([metrics.roc_auc_score(samples['true'][k], samples['prob'][k], average='macro', multi_class='ovr') for k in range(self.k_fold)])
            f1_score= aggregate([metrics.f1_score(samples['true'][k], samples['pred'][k], average='binary' if self.num_classes==2 else 'micro') for k in range(self.k_fold)])
            specificity = aggregate(self.specificity(samples['true'], samples['pred']))
            sensitivity = aggregate(self.sensitivity(samples['true'], samples['pred']))
        else:
            #如果进行了交叉验证，计算每个折的指标并根据 option 参数聚合结果
            accuracy = metrics.accuracy_score(samples['true'], samples['pred'])
            precision = metrics.precision_score(samples['true'], samples['pred'],pos_label=1, average='binary' if self.num_classes==2 else 'micro')
            recall = metrics.recall_score(samples['true'], samples['pred'], pos_label=1,average='binary' if self.num_classes == 2 else 'micro')
            f1_score= metrics.f1_score(samples['true'], samples['pred'],pos_label=1, average='binary' if self.num_classes==2 else 'micro')
            specificity = self.specificity(samples['true'], samples['pred'])
            sensitivity = self.sensitivity(samples['true'], samples['pred'])
            # precision = metrics.precision_score(samples['true'], samples['pred'], average='binary' if self.num_classes == 2 else 'micro')
            # recall = metrics.recall_score(samples['true'], samples['pred'], average='binary' if self.num_classes==2 else 'micro')
            # f1_score= metrics.f1_score(samples['true'], samples['pred'], average='binary' if self.num_classes==2 else 'micro')
            roc_auc = metrics.roc_auc_score(samples['true'], samples['prob'][:,1]) if self.num_classes==2 else metrics.roc_auc_score(samples['true'], samples['prob'], average='macro', multi_class='ovr')
        if initialize:
            self.initialize(k)
        # return dict(accuracy=accuracy, precision=precision, recall=recall, f1_score=f1_score)
        return dict(accuracy=accuracy, spesitivity=specificity, sensitivity=sensitivity, roc_auc=roc_auc)
        # return dict(accuracy=accuracy, precision=precision, recall=recall, roc_auc=roc_auc,f1_score=f1_score)
    def to_csv(self, targetdir, k=None, initialize=False):
        #将性能指标写入 CSV 文件
        metric_dict = self.evaluate(k, initialize) 
        append = os.path.isfile(os.path.join(targetdir, 'metric.csv'))#检查目标目录下是否存在名为 metric.csv 的文件。如果文件存在，append 变量将被设置为 True，表示后续写入时应该追加内容而不是覆盖原有内容.
        with open(os.path.join(targetdir, 'metric.csv'), 'a', newline='') as f:
            writer = csv.writer(f) 
            if not append:#如果文件是新创建的（not append）,则写入表头。表头包括 'fold' 和 metric_dict.keys() 中的键名。
                writer.writerow(['fold'] + [str(key) for key in metric_dict.keys()]) 
            writer.writerow([str(k)]+[str(value) for value in metric_dict.values()])
            #写入当前折的性能指标。如果 k 是 None，则表示不是在 K 折交叉验证中，而是在写入整体的性能指标。
            if k is None:
                writer.writerow([str(k)]+list(self.evaluate(k, initialize, 'std').values()))
