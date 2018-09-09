
import tensorflow as tf
from sklearn import cross_validation
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
from keras.preprocessing import image
import keras

import random as rn
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_json
from keras.applications.vgg16 import VGG16, preprocess_input #https://stackoverflow.com/questions/47555829/preprocess-input-method-in-keras

from PIL import Image
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
from sklearn.decomposition import RandomizedPCA
from sklearn.externals import joblib
from sklearn.svm import LinearSVC

from PIL import ImageFile
import math

ImageFile.LOAD_TRUNCATED_IMAGES = True

config = tf.ConfigProto(device_count={'GPU':1,'CPU':56})
sess = tf.Session(config=config)
keras.backend.set_session(sess)

STANDARD_SIZE = (300, 300)

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)
rn.seed(0)

#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(0)

input_shape = (300, 300, 3)
batch_size = 4
epochs = 10
num_classes = 2

f_log = './logs'
f_model = './model'


model_filename = 'cnn_model.json'
weights_filename = 'cnn_model_weights.hdf5'


def input_data(path_train, path_test):

    x = []

    with open(path_train, "r") as f:
        train_path_list = f.readlines()

    filenames = []
    labels = []
    for row in train_path_list:
        row = row.split(" ")
        filenames.append(row[0])
        labels.append(row[1])

    label = []
    for i in labels:
        label.append(int(i))

    for f in filenames:
        x.append(image.img_to_array(img_to_matrix(f)))

    x = np.asarray(x)
    #正規化
    x /= 255
    y = np.asarray(label)
    y = keras.utils.to_categorical(y, num_classes)

    '==============================================================='

    a = []

    with open(path_test, "r") as f:
        train_path_list = f.readlines()

    filenames1 = []
    labels1 = []
    for row in train_path_list:
        row = row.split(" ")
        filenames1.append(row[0])
        labels1.append(row[1])

    label1 = []
    for i in labels1:
        label1.append(int(i))

    for f in filenames1:
        a.append(image.img_to_array(img_to_matrix(f)))

    test_data = np.asarray(a)
    # 正規化
    test_data /= 255
    test_label = np.asarray(label1)
    test_label = keras.utils.to_categorical(test_label, num_classes)

    train_data, valid_data, train_label, valid_label = cross_validation.train_test_split(x, y, test_size=0.3)
    test_data, valid1, test_label, valid2 = cross_validation.train_test_split(test_data, test_label, test_size=0.3)

    return train_data, test_data, train_label, test_label, valid_data, valid_label



# parse image
def img_to_matrix(filename, verbose=False):
    img = Image.open(filename)
    if verbose:
        print('changing size from %s to %s' % (str(img.size), str(STANDARD_SIZE)))
    img = img.resize(STANDARD_SIZE)

    imgArray = np.asarray(img)
    return imgArray  # imgArray.shape = (167 x 300 x 3)


# 1次元に引き延ばす(PCAで使用)
def flatten_image(img):

    s = img.shape[0] * img.shape[1] * img.shape[2]
    img_wide = img.reshape(1, s)
    return img_wide[0]


def handle_image_with_pca(intermediate_output, y_test):
    images = intermediate_output
    labels = y_test
    ls = []

    for i in labels:
        if i == 1:
            ls.append("cloudy_seesaa")
        elif i == 0:
            ls.append("sunny_seesaa")
    labels = ls

    data = []
    for image in images:
        img = flatten_image(image)
        data.append(img)

    data = np.array(data)

    is_train = np.random.uniform(0, 1, len(data)) <= 0.7
    y = np.where(np.array(labels) == 'cloudy_seesaa', 1, 0)

    train_x, train_y = data[is_train], y[is_train]

    # plot in 2 dimensions
    pca = RandomizedPCA(n_components=2)
    X = pca.fit_transform(data)
    df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1],
                       "label": np.where(y == 1, 'cloudy_seesaa', 'sunny_seesaa')})
    colors = ['blue', 'red']

    plt.figure(figsize=(10, 10))
    for label, color in zip(df['label'].unique(), colors):
        mask = df['label'] == label
        plt.scatter(df[mask]['x'], df[mask]['y'], c=color, label=label)
    sns.set()
    plt.xlabel("pc1 (Principal Component1)")  # 全データの分散が最大となる方向
    plt.ylabel("pc2 (Principal Component2)")  # 第一主成分に垂直な方向の軸
    plt.legend()
    #plt.show()
    plt.savefig('pca_feature1.png')

    # training a classifier
    pca = RandomizedPCA(n_components=5)
    train_x = pca.fit_transform(train_x)

    svm = LinearSVC(C=1.0)
    svm.fit(train_x, train_y)
    joblib.dump(svm, 'model.pkl')

    # evaluating the model
    test_x, test_y = data[is_train == False], y[is_train == False]
    test_x = pca.transform(test_x)
    print(pd.crosstab(test_y, svm.predict(test_x),
                      rownames=['Actual'], colnames=['Predicted']))


def main():

    x_train, x_test, y_train, y_test, valid_data, valid_label = input_data("path_and_label_train.txt",
                                                                           "path_and_label_test.txt")

    old_session = KTF.get_session()

    with tf.Graph().as_default():

        session = tf.Session('')
        KTF.set_session(session)
        KTF.set_learning_phase(1)
        """
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape, kernel_initializer=keras.initializers.he_normal(),
                         bias_initializer="zeros"))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(28, activation='relu', init='he_uniform'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax')) # num_classes = 2値分類


        """
        # load trained model
        json_string = open(os.path.join(f_model, model_filename)).read()
        model = model_from_json(json_string)
        model.load_weights(os.path.join(f_model, weights_filename))
        """
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer="SGD",
                      metrics=['accuracy'])

        print(model.summary())

        # callback function
        tb_cb = keras.callbacks.TensorBoard(log_dir=f_log, histogram_freq=1)
        cbks = [tb_cb]

        # train
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,       #進行状況の表示モード
                            callbacks=cbks,  # [plot_losses, csv_logger],
                            validation_data=(x_test, y_test))
        score_train = model.evaluate(x_train,y_train, verbose=1, batch_size=4) 
        score_test = model.evaluate(x_test, y_test, verbose=1, batch_size=4)
        print('Train loss: {0}'.format(score_train[0]))
        print('Train accuracy: {0}'.format(score_train[1]))
        print('Test loss: {0}'.format(score_test[0]))
        print('Test accuracy: {0}'.format(score_test[1]))

        # 学習済みモデル書き出し
        json_string = model.to_json()
        open(os.path.join(f_model, 'cnn_model.json'), 'w').write(json_string)
        yaml_string = model.to_yaml()
        open(os.path.join(f_model, 'cnn_model.yaml'), 'w').write(yaml_string)
        print('save weights')
        model.save_weights(os.path.join(f_model, 'cnn_model_weights.hdf5'))
        """
        # modelのlayer_nameを調べる
        for layer in model.layers:
            print(layer.name)

        layer_name ="conv2d_2"#"dropout_1"#"conv2d_2"# "max_pooling2d_1"
        intermediate_layer_model = keras.models.Model(inputs=model.input,
                                         outputs=model.get_layer(layer_name).output)

       
        layers = model.layers[1:2]#[3:4]#[1:2]#[2:3]
        img = image.load_img("/home/seimei/Graduation_Research/dataset_valid/hare/class3-1/image_0064.jpg", target_size=(300, 300))
        img = image.img_to_array(img)
        img /= 255
        img = np.expand_dims(img, axis=0)
        # 指定したlayer_nameと一致するレイヤーの出力を取得する
        activations = intermediate_layer_model.predict(img)
        activations = [activation for layer, activation in zip(layers, activations) if isinstance(layer, Conv2D)]
        print(activations)
        # 単品の特徴画像生成#
        for i, activation in enumerate(activations):
            num_of_image = activation.shape[2]
            max = np.max(activation[0])
            for j in range(0, num_of_image):
                plt.figure()
                #array_img = activation[:,:,j]
                #print(np.argmax(array_img, axis=1))
                #im = Image.fromarray(np.uint8(matplotlib.cm.gist_earth(array_img)*255))
                #ima = Image.fromarray(array_img.astype('uint8'), 'RGB')
                #plt.imshow(im)
                sns.heatmap(activation[:, :,j], vmin=0, vmax=max, xticklabels=False, yticklabels=False, square=False)
                plt.savefig("%d_%d.png" % (i+1, j+1))
                plt.close()
        # 出力層ごとに特徴画像を並べてヒートマップ画像として出力
        for i, activation in enumerate(activations):
            num_of_image = activation.shape[2]
            cols = math.ceil(math.sqrt(num_of_image))
            rows = math.floor(num_of_image / cols)
            screen = []
            for y in range(0, rows):
                row = []
                for x in range(0, cols):
                    j = y * cols + x
                    if j < num_of_image:
                        row.append(activation[:, :, j])
                    else:
                        row.append(np.zeros())
                screen.append(np.concatenate(row, axis=1))
            screen = np.concatenate(screen, axis=0)
            plt.figure()
            sns.heatmap(screen, xticklabels=False, yticklabels=False)
            name = "maxpooling2d"
            plt.savefig("%s.png" % name)
            plt.close()
       
        handle_image_with_pca(activations, np.zeros(1))


    

    # add for TeonsorBoard

    KTF.set_session(old_session)


if __name__ == '__main__':
    main()
