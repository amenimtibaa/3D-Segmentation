import numpy as np
import keras
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Reshape
from keras.layers import Convolution1D, BatchNormalization, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import Lambda, concatenate
import tensorflow as tf


def extend_dimension(global_feature, axis):
    '''
    Extend dimension of a tensor(example: [None, 1024] to [None, 1, 1024])
    '''
    return tf.expand_dims(global_feature, axis)


def extend_size(global_feature, num_points):
    '''
    Extend size of a tensor(example: [None, 1, 1024] to [None, num_points, 1024])
    '''
    return tf.tile(global_feature, [1, num_points, 1])


def multilayer_perceptron(inputs, mlp_nodes):
    '''
    Define multilayer-perceptron
    '''
    mlp = []
    x = inputs
    for i, num_nodes in enumerate(mlp_nodes):
        x = Convolution1D(filters=num_nodes, kernel_size=1, activation='relu')(x)
        x = BatchNormalization()(x)
        mlp.append(x)
    return mlp


def TNet(inputs, tsize, mlp_nodes=(64, 128, 1024), fc_nodes=(512, 256)):
    '''
    Define T-Net to predict affine transformation matrix
    '''
    x = inputs
    for i, num_nodes in enumerate(mlp_nodes):
        x = Convolution1D(filters=num_nodes, kernel_size=1, activation='relu')(x)
        x = BatchNormalization()(x)

    x = GlobalMaxPooling1D()(x)

    for i, num_nodes in enumerate(fc_nodes):
        x = Dense(num_nodes, activation='relu')(x)
        x = BatchNormalization()(x)

    x = Dense(tsize * tsize,
              weights=[np.zeros([num_nodes, tsize * tsize]), np.eye(tsize).flatten().astype(np.float32)])(
        x)  # constrain the feature transformation matrix to be close to orthogonal matrix
    transformation_matrix = Reshape((tsize, tsize))(x)
    return transformation_matrix


def PointNetFull(num_points, num_classes):
    '''
    Pointnet full architecture
    :param num_points: an integer that is the number of input points
    :param num_classes: an integer that is number of categories
    '''

    inputs = Input(shape=(num_points, 3))

    tnet1 = TNet(inputs=inputs, tsize=3, mlp_nodes=(128, 128, 1024), fc_nodes=(512, 256))
    aligned_feature1 = keras.layers.dot(inputs=[inputs, tnet1], axes=2)

    extracted_feature11, extracted_feature12, extracted_feature13 = multilayer_perceptron(inputs=aligned_feature1,
                                                                                          mlp_nodes=(64, 128, 128))

    tnet2 = TNet(inputs=inputs, tsize=128, mlp_nodes=(128, 128, 1024), fc_nodes=(512, 256))
    aligned_feature2 = keras.layers.dot(inputs=[extracted_feature13, tnet2], axes=2)

    extracted_feature21, extracted_feature22 = multilayer_perceptron(inputs=aligned_feature2, mlp_nodes=(512, 2048))

    global_feature = GlobalMaxPooling1D()(extracted_feature22)

    global_feature_seg = Lambda(extend_dimension, arguments={'axis': 1})(global_feature)
    global_feature_seg = Lambda(extend_size, arguments={'num_points': num_points})(global_feature_seg)

    # Segmentation block
    seg = concatenate(
        [extracted_feature11, extracted_feature12, extracted_feature13, aligned_feature2, extracted_feature21,
         global_feature_seg])
    seg1, seg2, seg3 = multilayer_perceptron(inputs=seg, mlp_nodes=(256, 256, 128))
    segmentation_result = Convolution1D(num_classes, 1, padding='same', activation='softmax')(seg3)

    model = Model(inputs=inputs, outputs=segmentation_result)
    return model



if __name__ == '__main__':
    num_points = 1024
    batch_size = 36
    num_epoches = 50
    cat_choices = 'Airplane'

    """
    #test model

    model = PointNetFull(num_points=num_points, num_classes=num_classes)
    model.compile(optimizer= ... ,
                  loss= ...,
                  metrics= ...)
    """