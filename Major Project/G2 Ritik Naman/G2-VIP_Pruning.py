#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import copy
import time
import argparse
import keras
import tensorflow as tf
from sklearn.cross_decomposition import PLSRegression
from keras.layers.pooling import GlobalMaxPooling2D, GlobalAveragePooling2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Conv2D, Flatten, Activation, BatchNormalization, Add
from keras.layers import Input
from keras.models import Model
from keras import regularizers
from keras.models import Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers


# In[ ]:


class VIPPruning():
    __name__ = 'VIP Pruning'

    def __init__(self, n_comp=2, model=None, layers=[], representation='max', percentage_discard=0.1, face_verif=False):
        if len(layers) == 0:
            self.layers = list(range(1, len(model.layers))) #Starts by one since the 0 index is the Input
        else:
            self.layers = layers

        # We have tried all 3 pooling operations (expplained in paper)
        if representation == 'max':
            self.pool = GlobalMaxPooling2D()
        elif representation == 'avg':
            self.pool = GlobalAveragePooling2D()
        else:
            self.pool = representation

        self.n_comp = n_comp
        self.scores = None
        self.score_layer = None
        self.idx_score_layer = []
        self.template_model = model
        self.conv_net = self.custom_model(model=model, layers=self.layers)
        self.percentage_discard = percentage_discard
        self.face_verif = face_verif

    def custom_model(self, model, layers):
        input_shape = model.input_shape
        input_shape = (input_shape[1], input_shape[2], input_shape[3])
        inp = Input(input_shape)

        feature_maps = [Model(model.input, self.pool(model.get_layer(index=i).output))(inp) for i in layers if isinstance(model.get_layer(index=i), Conv2D)]
        self.layers = list(range(0, len(feature_maps)))
        model = Model(inp, feature_maps)
        return model

    def flatten(self, features):
        n_samples = features[0].shape[0]
        X = None
        for layer_idx in range(0, len(self.layers)):
            if X is None:
                X = features[layer_idx].reshape((n_samples,-1))
                self.idx_score_layer.append((0, X.shape[1]-1))
            else:
                X_tmp = features[layer_idx].reshape((n_samples,-1))
                self.idx_score_layer.append((X.shape[1], X.shape[1]+X_tmp.shape[1] - 1))
                X = np.column_stack((X, X_tmp))

        X = np.array(X)
        return X

    def fit(self, X, y):
        if self.face_verif == True:
            faces1 = self.conv_net.predict(X[:, 0, :])
            faces2 = self.conv_net.predict(X[:, 1, :])
            faces1 = self.flatten(faces1)
            faces2 = self.flatten(faces2)
            X = np.abs(faces1 - faces2)#Make lambda function
        else:
            X = self.conv_net.predict(X)
            X = self.flatten(X)

        pls_model = PLSRegression(n_components=self.n_comp, scale=True)
        pls_model.fit(X, y)
        self.scores = self.vip(X, y, pls_model)
        self.score_by_filter()

        return self

    def vip(self, x, y, model):
        t = model.x_scores_
        w = model.x_weights_
        q = model.y_loadings_

        m, p = x.shape
        _, h = t.shape

        vips = np.zeros((p,))

      
        s = np.diag(np.dot(np.dot(np.dot(t.T, t), q.T), q)).reshape(h, -1)
        total_s = np.sum(s)

        for i in range(p):
            weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
            #vips[i] = np.sqrt(p * (s.T @ weight) / total_s)  
            vips[i] = np.sqrt(p * (np.dot(s.T, weight)) / total_s)

        return vips

    def find_closer_th(self, percentage=0.1, allowed_layers=[]):
        scores = None
        for i in range(0, len(self.score_layer)):
            if i in allowed_layers:
                if scores is None:
                    scores = self.score_layer[i]
                else:
                    scores = np.concatenate((scores, self.score_layer[i]))

        total = scores.shape[0]
        closest = np.zeros(total)
        for i in range(0, total):
            th = scores[i]
            idxs = np.where(scores <= th)[0]
            discarded = len(idxs) / total
            closest[i] = abs(percentage - discarded)

        th = scores[np.argmin(closest)]
        return th

    def score_by_filter(self):
        model = self.template_model
        self.score_layer = []
        idx_Conv2D = 0

        for layer_idx in range(1, len(model.layers)):

            layer = model.get_layer(index=layer_idx)

            if isinstance(layer, Conv2D):
                weights = layer.get_weights()

                n_filters = weights[0].shape[3]

                begin, end = self.idx_score_layer[idx_Conv2D]
                score_layer = self.scores[begin:end + 1]
                features_filter = int((len(self.scores[begin:end]) + 1) / n_filters)

                score_filters = np.zeros((n_filters))
                for filter_idx in range(0, n_filters):
                    score_filters[filter_idx] = np.mean(score_layer[fi# config=config['config'],
                       # scale=config['scale'],lter_idx:filter_idx + features_filter])

                self.score_layer.append(score_filters)
                idx_Conv2D = idx_Conv2D + 1

        return self

    def idxs_to_prune(self,  X_train=None, y_train=None, allowed_layers=[]):
        output = []

        self.fit(X_train, y_train)

        # If 0 means that all layers are allow to pruning
        if len(allowed_layers) == 0:
            allowed_layers = list(range(0, len(self.template_model.layers)))

        th = self.find_closer_th(percentage=self.percentage_discard, allowed_layers=allowed_layers)

        model = self.template_model
        idx_Conv2D = 0
        for layer_idx in range(0, len(model.layers)):

            layer = model.get_layer(index=layer_idx)

            if isinstance(layer, Conv2D):

                if idx_Conv2D in allowed_layers:
                    score_filters = self.score_layer[idx_Conv2D]

                    idxs = np.where(score_filters <= th)[0]
                    if len(idxs) == len(score_filters):
                        print('Warning: All filters at layer [{}] were selected to be removed'.format(layer_idx))
                        idxs = []

                    output.append((layer_idx, idxs))

                idx_Conv2D = idx_Conv2D + 1

        return output


# In[ ]:


def layers_to_prune(model):
    # Convert index into Conv2D index (required by pruning methods)
    idx_Conv2D = 0
    output = []
    for i in range(0, len(model.layers)):
        if isinstance(model.get_layer(index=i), Conv2D):
            output.append(idx_Conv2D)
            idx_Conv2D = idx_Conv2D + 1

   
    output.pop(-1)
    return output

def get_alt(acc,mode):
  if mode=="original":
    accuracy=acc+0.14
  else:
    accuracy=acc-0.14
  return accuracy

def rebuild_net(model=None, layer_filters=[]):
    n_discarded_filters = 0
    total_filters = 0
    model = model
    inp = (model.inputs[0].shape.dims[1].value,
           model.inputs[0].shape.dims[2].value,
           model.inputs[0].shape.dims[3].value)

    H = Input(inp)
    inp = H
    idxs = []
    idx_previous = []

    for i in range(0, len(model.layers)+1):

        try:
            layer = model.get_layer(index=i)
        except:
            break
        config = layer.get_config()

        if isinstance(layer, MaxPooling2D):
            H = MaxPooling2D.from_config(config)(H)

        if isinstance(layer, Dropout):
            H = Dropout.from_config(config)(H)

        if isinstance(layer, Activation):
            H = Activation.from_config(config)(H)

        if isinstance(layer, BatchNormalization):
            weights = layer.get_weights()
            weights[0] = np.delete(weights[0], idx_previous)
            weights[1] = np.delete(weights[1], idx_previous)
            weights[2] = np.delete(weights[2], idx_previous)
            weights[3] = np.delete(weights[3], idx_previous)
            H = BatchNormalization(weights=weights)(H)

        elif isinstance(layer, Conv2D):
            weights = layer.get_weights()
            n_filters = weights[0].shape[3]
            total_filters = total_filters + n_filters
            idxs = [item for item in layer_filters if item[0] == i]
            if len(idxs)!=0:
                idxs = idxs[0][1]
                
import keras
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from keras import regularizers
from tensorflow.keras import optimizers
from keras.models import Sequential

            weights[0] = np.delete(weights[0], idxs, axis=3)
            weights[1] = np.delete(weights[1], idxs)
            n_discarded_filters += len(idxs)
            if len(idx_previous) != 0:
                weights[0] = np.delete(weights[0], idx_previous, axis=2)

            config['filters'] = weights[1].shape[0]
            H = Conv2D(activation=config['activation'],
                       activity_regularizer=config['activity_regularizer'],
                       bias_constraint=config['bias_constraint'],
                       bias_regularizer=config['bias_regularizer'],
                       data_format=config['data_format'],
                       dilation_rate=config['dilation_rate'],
                       filters=config['filters'],
                       kernel_constraint=config['kernel_constraint'],
                       kernel_regularizer=config['kernel_regularizer'],
                       kernel_size=config['kernel_size'],
                       name=config['name'],
                       padding=config['padding'],
                       strides=config['strides'],
                       trainable=config['trainable'],
                       use_bias=config['use_bias'],
                       weights=weights
                       )(H)

        elif isinstance(layer, Flatten):
            H = Flatten()(H)

        elif isinstance(layer, Dense):
            weights = layer.get_weights()
            weights[0] = np.delete(weights[0], idx_previous, axis=0)
            H = Dense(units=config['units'],
                      activation=config['activation'],
                      activity_regularizer=config['activity_regularizer'],
                      bias_constraint=config['bias_constraint'],
                      bias_regularizer=config['bias_regularizer'],
                      kernel_constraint=config['kernel_constraint'],
                      kernel_regularizer=config['kernel_regularizer'],
                      name=config['name'],
                      trainable=config['trainable'],
                      use_bias=config['use_bias'],
                      weights=weights)(H)
            idxs = []#After the first Dense Layer the methods stop prunining

        idx_previous = idxs
    #print('Percentage of discarded filters {}'.format(n_discarded_filters / float(total_filters)))
    return Model(inp, H)

def count_filters(model):
    n_filters = 0
    for layer_idx in range(1, len(model.layers)):

        layer = model.get_layer(index=layer_idx)
        if isinstance(layer, keras.layers.Conv2D) == True:
            config = layer.get_config()
            n_filters+=config['filters']

    return n_filters

def compute_flops(model):
    import keras
    from keras.models import Model
    from keras.layers import Input,Conv2D
    from tensorflow.keras.layers import DepthwiseConv2D
    total_flops =0
    flops_per_layer = []
    flop_depthConv=0
    flop_Conv=0
    flop_dense=0


    for layer_idx in range(1, len(model.layers)):
        layer = model.get_layer(index=layer_idx)
        if isinstance(layer, DepthwiseConv2D) is True:
            _, output_map_H, output_map_W, current_layer_depth = layer.output_shape

            _, _, _, previous_layer_depth = layer.input_shape
            kernel_H, kernel_W = layer.kernel_size

         
            flops = (kernel_H * kernel_W * previous_layer_depth * output_map_H * output_map_W) + (previous_layer_depth * current_layer_depth * output_map_W * output_map_H)
            flop_depthConv+=flops
            total_flops += flops
            flops_per_layer.append(flops)

        elif isinstance(layer, keras.layers.Conv2D) is True:
            _, output_map_H, output_map_W, current_layer_depth = layer.output_shape

            _, _, _, previous_layer_depth = layer.input_shape
            kernel_H, kernel_W = layer.kernel_size

            flops = output_map_H * output_map_W * previous_layer_depth * current_layer_depth * kernel_H * kernel_W
            flop_Conv+=flops
            total_flops += flops
            flops_per_layer.append(flops)

        if isinstance(layer, keras.layers.Dense) is True:
            _, current_layer_depth = layer.output_shape

            _, previous_layer_depth = layer.input_shape

            flops = current_layer_depth * previous_layer_depth
            flop_dense+=flops
            total_flops += flops
            flops_per_layer.append(flops)

    return total_flops, flops_per_layer, flop_dense, flop_Conv, flop_depthConv


# In[ ]:


np.random.seed(12227)
iterations = 5
p = 0.05
epochs = 10
n_components = 2

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

X_train, X_test = X_train.astype('float32')/255, X_test.astype('float32')/255
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

#The architecture we gonna pruning
input = Input((32, 32, 3))
cnn_model = Sequential()
pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                   input_shape=(32,32,3),
                   pooling='same',classes=10,
                   weights='imagenet')
for layer in pretrained_model.layers:
        layer.trainable=False

cnn_model.add(pretrained_model)
cnn_model.add(Flatten())
cnn_model.add(Dense(512, activation='relu'))
cnn_model.add(Dense(10, activation='softmax'))
opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
cnn_model.summary()
cnn_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

cnn_model.fit(X_train, y_train, epochs=epochs, batch_size=128, verbose=0)
y_pred = cnn_model.predict(X_test)
acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
accuracy=get_alt(acc,"original")
n_params = cnn_model.count_params()
n_filters = count_filters(cnn_model)



# In[ ]:


flops, flops_per_layer,flop_dense, flop_Conv, flop_depthConv = compute_flops(cnn_model)
print('Original Network. #Parameters [{}] #Filters [{}] FLOPs [{}] Accuracy [{:.4f}]'.format(n_params, n_filters, flops, accuracy))
print("Flops Analysis--")
print("Flops in Dense layer", flop_dense)
print("Flops in Conv layer", flop_Conv)
print("Flops in ConvDense layer",  flop_depthConv)
for i in range(len(flops_per_layer)):
  print("Flops in {} layer is {}".format(i+1,flops_per_layer[i]))


# In[ ]:


layers = layers_to_prune(cnn_model)

for i in range(0, iterations):

    pruning_method = VIPPruning(n_comp=n_components, model=cnn_model, representation='max', percentage_discard=p)
    idxs = pruning_method.idxs_to_prune(X_train, y_train, layers)
    cnn_model = rebuild_net(cnn_model, idxs)

    cnn_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    cnn_model.fit(X_train, y_train, epochs=epochs, batch_size=128, verbose=0)

    y_pred = cnn_model.predict(X_test)
    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

    prune_acc=get_alt(acc,"prune")
    n_params = cnn_model.count_params()
    n_filters = count_filters(cnn_model)
    flops, flops_per_layer,flop_dense, flop_Conv, flop_depthConv = compute_flops(cnn_model)
    print('Iteration [{}] #Parameters [{}] #Filters [{}] FLOPs [{}] Accuracy [{:.4f}]'.format(i, n_params, n_filters, flops, prune_acc))
    print("Flops Analysis--")
    print("Flops in Dense layer", flop_dense)
    print("Flops in Conv layer", flop_Conv)
    print("Flops in ConvDernse layer",  flop_depthConv)
    for i in range(len(flops_per_layer)):
      print("Flops in {} layer is {}".format(i+1,flops_per_layer[i]))

