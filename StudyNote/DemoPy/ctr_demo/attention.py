import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization,Dropout,Dense,Flatten,Conv1D
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers
import time

try:
    from tensorflow.python.keras.callbacks import CallbackList
except ImportError:
    from tensorflow.python.keras._impl.keras.callbacks import CallbackList
    
def slice_arrays(arrays, start=None, stop=None):
    """Slice an array or list of arrays.

    This takes an array-like, or a list of
    array-likes, and outputs:
        - arrays[start:stop] if `arrays` is an array-like
        - [x[start:stop] for x in arrays] if `arrays` is a list

    Can also work on list/array of indices: `slice_arrays(x, indices)`

    Arguments:
        arrays: Single array or list of arrays.
        start: can be an integer index (start index)
            or a list/array of indices
        stop: integer (stop index); should be None if
            `start` was a list.

    Returns:
        A slice of the array(s).

    Raises:
        ValueError: If the value of start is a list and stop is not None.
    """

    if arrays is None:
        return [None]

    if isinstance(arrays, np.ndarray):
        arrays = [arrays]

    if isinstance(start, list) and stop is not None:
        raise ValueError('The stop argument has to be None if the value of start '
                         'is a list.')
    elif isinstance(arrays, list):
        if hasattr(start, '__len__'):
            # hdf5 datasets only support list objects as indices
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [None if x is None else x[start] for x in arrays]
        else:
            if len(arrays) == 1:
                return arrays[0][start:stop]
            return [None if x is None else x[start:stop] for x in arrays]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return arrays[start]
        elif hasattr(start, '__getitem__'):
            return arrays[start:stop]
        else:
            return [None]


class BaseModel(nn.Module):
    def __init__(self, device='cpu'):
        super(BaseModel, self).__init__()
        
        # device
        self.device = device 
        
        self.to(device)
    
    def compile(self, optimizer,
                loss=None,
                metrics=None,
                ):
        """
        :param optimizer: String (name of optimizer) or optimizer instance. See [optimizers](https://pytorch.org/docs/stable/optim.html).
        :param loss: String (name of objective function) or objective function. See [losses](https://pytorch.org/docs/stable/nn.functional.html#loss-functions).
        :param metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']`.
        """
        self.metrics_names = ["loss"]
        self.optim = self._get_optim(optimizer)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)

    def _get_optim(self, optimizer):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=0.01)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters())  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters())  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters())
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            if loss == "binary_crossentropy":
                loss_func = F.binary_cross_entropy
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss
        return loss_func

    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        # change eps to improve calculation accuracy
        return log_loss(y_true,
                        y_pred,
                        eps,
                        normalize,
                        sample_weight,
                        labels)

    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                        y_true, np.where(y_pred > 0.5, 1, 0))
                self.metrics_names.append(metric)
        return metrics_
    
    def fit(self, x = None, y = None, batch_size=None, epochs=1, verbose=2,  validation_split=0.1, shuffle=True, callbacks=None):
        
        if validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
                
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))
        else:
            do_validation = False
            val_x = []
            val_y = []
        
        train_tensor_data = Data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        
        if batch_size is None:
            batch_size = 256
            
        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)
        
        print(self.device, end="\n")
        model = self.train()
        loss_func = self.loss_func
        optim = self.optim
        
        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        self.stop_training = False  # used for early stopping

        # Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        
        for epoch in range(epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            try:
                with tqdm(enumerate(train_loader), disable = verbose != 1) as t:
                    for index, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()

                        y_pred = model(x).squeeze()
                        #---------------------------
                        #print(y_pred.shape)
                        
                        optim.zero_grad()
                        loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                        
                        #reg_loss = self.get_regularization_loss()

                        #total_loss = loss + reg_loss + self.aux_loss
                        total_loss = loss
                        
                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        total_loss.backward(retain_graph=True)
                        
                        optim.step()

                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name not in train_result:
                                    train_result[name] = []
                                train_result[name].append(metric_fun(
                                    y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))

            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
                    
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])

                for name in self.metrics:
                    eval_str += " - " + name + \
                                ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += " - " + "val_" + name + \
                                    ": {0: .4f}".format(epoch_logs["val_" + name])
                print(eval_str)
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()
        
    def evaluate(self, x, y, batch_size=256):
        """

        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        pred_ans = self.predict(x, batch_size)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            eval_result[name] = metric_fun(y, pred_ans)
        return eval_result

    def predict(self, x, batch_size=256):
        """

        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()
        tensor_data = Data.TensorDataset(torch.tensor(x))
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)

        pred_ans = []
        with torch.no_grad():
            for index, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()

                y_pred = model(x).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)

        return np.concatenate(pred_ans).astype("float64")

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim = 32, num_heads = 8, device = 'cpu'):
        super(MultiHeadAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
            
        self.projection_dim = embed_dim // num_heads
        
        self.query_dense = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_dense = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_dense = nn.Linear(embed_dim, embed_dim, bias=False)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        
        self.to(device)
        
    def attention(self, query, key, value):
        score = torch.matmul(query, key.transpose(-1, -2))
        dim_key = torch.tensor(key.shape[-1], dtype = torch.float32)
        scaled_score = score / torch.sqrt(dim_key)
        weights = self.softmax(scaled_score)
        output = torch.matmul(weights, value)
        return output, weights
    
    def separate_heads(self, x, batch_size):
        x = x.reshape((batch_size, -1, self.num_heads, self.projection_dim))
        return x.permute(0,2,1,3)
    
    def forward(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = inputs.shape[0]
        
        # (batch_size, seq_len, embed_dim)
        query = self.query_dense(inputs) 
        
        # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  
        
        # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)
        
        # (batch_size, num_heads, seq_len, projection_dim)
        query = self.separate_heads(
            query, batch_size
        )  
        
        # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  
        
         # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        ) 
        attention, weights = self.attention(query, key, value)
        
        # (batch_size, seq_len, num_heads, projection_dim)
        attention = attention.permute(0, 2, 1, 3) 
        
        # (batch_size, seq_len, embed_dim)
        concat_attention = attention.reshape(batch_size, -1, self.embed_dim)  
        
         # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(concat_attention) 
        return output


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1, device = 'cpu'):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.to(device)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn



class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, embed_dim = 32, num_heads = 8, device = 'cpu', dropout=0.1):
        super().__init__()

        self.n_head = num_heads
        self.d_k = embed_dim // num_heads
        self.d_v = embed_dim // num_heads
        d_model = embed_dim
        
        self.w_qs = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_ks = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_vs = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fc = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        self.to(device)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, maxlen, vocab_size, embed_dim, device = 'cpu'):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = nn.Embedding(num_embeddings  = vocab_size, embedding_dim  = embed_dim)
        self.pos_emb = nn.Embedding(num_embeddings = maxlen, embedding_dim = embed_dim)
        self.device = device
        self.to(device)
    
    # 这里默认输入的是tensor
    def forward(self, x):
        maxlen = x.shape[-1]
        positions = torch.arange(0, maxlen).to(self.device)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
    
class TransformerBlock(BaseModel):
    def __init__(self, embed_dim = 32 , num_heads = 8 , ff_dim = 64, dropout = 0.2, device = 'cuda:0'):
        super(TransformerBlock, self).__init__(device = device)
        self.embedding_layer = TokenAndPositionEmbedding(maxlen = 32, vocab_size = 50000, embed_dim = 32, device = device)

        self.att = MultiHeadAttention(embed_dim = 32, num_heads = 8, device = device)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(inplace = True),
            nn.Linear(ff_dim, embed_dim)
        )
        
        self.layernorm1 = nn.LayerNorm(embed_dim, eps = 1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps = 1e-6)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, embed_dim))
        self.dropout3 = nn.Dropout(dropout)
        
        self.linear1 = nn.Linear(embed_dim, 20)
        self.relu1 = nn.ReLU(inplace = True)
        
        self.dropout4 = nn.Dropout(dropout)
        
        self.outlinear = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()
        
        self.device = device
        self.to(device)

    def forward(self, inputs):
        #print(inputs.shape)
        inputs = self.embedding_layer(inputs.type(torch.long))
        attn_output = self.att(inputs, inputs, inputs)
        
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        
        out = self.layernorm2(out1 + ffn_output)
        out = self.global_avg_pool(out)
        out = torch.flatten(out, start_dim=1)
        
        out = self.dropout3(out)
        
        out = self.linear1(out)
        out = self.relu1(out)
        out = self.dropout4(out)
        
        outputs = self.sigmoid(self.outlinear(out))
        
        return outputs    

'''
#test -------------
input_ = np.array([1,2,3,4,5,6,7,8,9]).reshape(3,3)
input_ = keras.preprocessing.sequence.pad_sequences(input_, maxlen=32)
input_ = torch.from_numpy(input_).type(torch.long)
#print(input_.shape)
test = TokenAndPositionEmbedding(50000, 10, 32)
a = test(input_)
#print(a.shape)
test_att = TransformerBlock()
a = test_att(a)
print(a.shape)
'''

maxlen = 32 
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)


# Embedding size for each token
embed_dim = 32 

# Number of attention heads
num_heads = 8  

# Hidden layer size in feed forward network inside transformer
ff_dim = 64  

vocab_size = 50000

model = TransformerBlock(embed_dim, num_heads, ff_dim)
model.compile('adam', 'binary_crossentropy',metrics=["binary_crossentropy", "auc"],)
model.fit(X_train, y.values, batch_size=20, epochs=100, validation_split=0.)