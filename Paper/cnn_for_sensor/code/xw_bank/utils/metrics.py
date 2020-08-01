import numpy as np

class XWMetrics(object):
    def __init__(self):
        self.labels = []
        self.scores = []
    
    def reset(self):
        self.labels = []
        self.scores = []
    

    def add_batch(self, labels, scores):
        self.labels.append(labels)
        self.scores.append(scores)
    
    def apply(self):
        labels         = np.concatenate(self.labels,axis=0)
        scores         = np.concatenate(self.scores,axis=0)
        acc_combo_func = get_acc_combo()
        acc_func       = get_acc_func()
        acc_combo      = acc_combo_func(labels, scores)
        acc            = acc_func(labels, scores)
        return {"acc":acc,"acc_combo":acc_combo}




def get_acc_combo():
    def combo(y, y_pred):
        # 数值ID与行为编码的对应关系
        mapping = {0: 'A_0', 1: 'A_1', 2: 'A_2', 3: 'A_3', 
            4: 'D_4', 5: 'A_5', 6: 'B_1',7: 'B_5', 
            8: 'B_2', 9: 'B_3', 10: 'B_0', 11: 'A_6', 
            12: 'C_1', 13: 'C_3', 14: 'C_0', 15: 'B_6', 
            16: 'C_2', 17: 'C_5', 18: 'C_6'}
        # 将行为ID转为编码
        code_y, code_y_pred = mapping[y], mapping[y_pred]
        if code_y == code_y_pred: #编码完全相同得分1.0
            return 1.0
        elif code_y.split("_")[0] == code_y_pred.split("_")[0]: #编码仅字母部分相同得分1.0/7
            return 1.0/7
        elif code_y.split("_")[1] == code_y_pred.split("_")[1]: #编码仅数字部分相同得分1.0/3
            return 1.0/3
        else:
            return 0.0

    confusionMatrix = np.zeros((19,19))
    for i in range(19):
        for j in range(19):
            confusionMatrix[i, j] = combo(i,j)
    
    def acc_combo(y,y_pred):
        #print(y_pred.size())
        y_pred = np.argmax(y_pred, axis = 1)
        scores = confusionMatrix[y.astype(np.int), y_pred.astype(np.int)]
        return np.mean(scores)
    
    return acc_combo

def get_acc_func():
    confusionMatrix=np.zeros((19,19))
    for i in range(19):
        confusionMatrix[i,i]=1
    def acc_func(y, y_pred):
        #y=np.argmax(y,axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        scores=confusionMatrix[y.astype(np.int),y_pred.astype(np.int)]
        return np.mean(scores)
    return acc_func
    


if __name__ == "__main__":
    metrics = XWMetrics()
