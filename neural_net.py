# Copyright [2020] Luis Alberto Pineda Cortés, Rafael Morales Gamboa.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import math
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, Dense, Flatten, \
    Reshape, Conv2DTranspose, BatchNormalization, LayerNormalization, SpatialDropout2D, \
    UpSampling2D, Activation
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from joblib import Parallel, delayed
from sklearn.utils import shuffle
import constants
import dataset

np.random.seed(constants.seed_value)
tf.random.set_seed(constants.seed_value)  # tf cpu fix seed
tf.keras.utils.set_random_seed(constants.seed_value)
tf.config.experimental.enable_op_determinism()

batch_size_ae  = 256
batch_size_cla = 32
epochs = 300
patience = 10
truly_training_percentage = 0.80

img_rows = 32
img_columns = 32

##
##
def get_encoder():
    domain = constants.domain
    
    input_img = Input(shape=(img_rows, img_columns, 3))
    # initializer = tf.keras.initializers.GlorotUniform()
    x = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu',
               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=constants.seed_value),
               input_shape=(32,32,3))(input_img) # 32 x 32 x 32
    x = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu',
               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=constants.seed_value))(x) # 32 x 32 x 16
    x = Conv2D(filters=8, kernel_size=3, padding='same', activation='relu',
               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=constants.seed_value))(x) # 32 x 32 x 8
    x = Conv2D(filters=4, kernel_size=3, padding='same', activation='relu',
               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=constants.seed_value))(x) # 32 x 32 x 4
    #
    if domain > 256:
        x = MaxPool2D((2,2), padding ='same')(x) # 16 x 16 x 4
        if domain < 1024:
            x = Conv2D(filters=2, kernel_size=3, padding='same', activation='relu',
                       kernel_initializer=tf.keras.initializers.GlorotUniform(seed=constants.seed_value))(x) # 16 x 16 x 2
        x = Flatten()(x)
    else:
        x = MaxPool2D((2,2), strides=(4,4), padding ='same')(x) # 8 x 8 x 4
        if domain < 256:
            x = Conv2D(filters=2, kernel_size=3, padding='same', activation='relu',
                       kernel_initializer=tf.keras.initializers.GlorotUniform(seed=constants.seed_value))(x) # 8 x 8 x 2
        x = Flatten()(x)
        if domain < 128:
            x = Dense(domain)(x)
    #
    x = LayerNormalization(name='encoded')(x)
    return input_img, x

##
##
def get_decoder():
    domain = constants.domain
    
    encoded_input = Input(shape=(domain, ))
    initializer = tf.keras.initializers.GlorotUniform()
    #
    if domain > 256:
        if domain < 1024:
            y = Reshape((16,16,2))(encoded_input)
            y = Conv2D(filters=4, kernel_size=3, padding='same', activation='relu',
                       kernel_initializer=tf.keras.initializers.GlorotUniform(seed=constants.seed_value))(y) # 16 x 16 x 4
        else:
            y = Reshape((16,16,4))(encoded_input) # 16 x 16 x 4
        y = UpSampling2D((2,2))(y) # 32 x 32 x 4
    else:
        if domain < 256:
            if domain < 128:
                y = Dense(128)(encoded_input)
                y = Reshape((8,8,2))(y) # 8 x 8 x 2
            else:
                y = Reshape((8,8,2))(encoded_input) # 8 x 8 x 2
                
            y = Conv2D(filters=4, kernel_size=3, padding='same', activation='relu',
                       kernel_initializer=tf.keras.initializers.GlorotUniform(seed=constants.seed_value))(y) # 8 x 8 x 4
        else:
            y = Reshape((8,8,4))(encoded_input) # 8 x 8 x 4
        y = UpSampling2D((4,4))(y) # 32 x 32 x 4
    #
    y = Conv2D(filters=8, kernel_size=3, padding='same', activation='relu',
               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=constants.seed_value))(y) # 32 x 32 x 8
    y = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu',
               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=constants.seed_value))(y) # 32 x 32 x 16
    y = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu',
               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=constants.seed_value))(y) # 32 x 32 x 32
    y = Conv2D(filters=3, kernel_size=3, activation='sigmoid',
               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=constants.seed_value), padding="same", name='decoded')(y) # 32x32x3
    return encoded_input, y

##
##
def get_classifier():
    width = img_columns
    domain = constants.domain
    
    encoded_input = Input(shape=(domain, ))
    weight_decay = 5e-4
    #initializer = tf.keras.initializers.GlorotUniform()
    ##
    c = Reshape((width//2, width//2, domain//256))(encoded_input) \
        if domain > 256 else Reshape((width//4, width//4, domain//64))(encoded_input)
    ##
    c = Conv2D(32, kernel_size=3, padding="same", activation='relu',
               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=constants.seed_value), kernel_regularizer=regularizers.l2(weight_decay))(c) # 16 x 16 x 32 | 8 x 8 x 32
    c = BatchNormalization()(c)
    c = Conv2D(32, kernel_size=3, padding="same", activation='relu',
               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=constants.seed_value), kernel_regularizer=regularizers.l2(weight_decay))(c) # 16 x 16 x 32 | 8 x 8 x 32
    c = BatchNormalization()(c)
    c = MaxPool2D(pool_size=(2,2), strides=(1,1), padding="same")(c) # 16 x 16 x 32 | 8 x 8 x 32
    c = Dropout(0.25)(c)
    ##
    c = Conv2D(64, kernel_size=3, padding="same", activation='relu',
               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=constants.seed_value), kernel_regularizer=regularizers.l2(weight_decay))(c) # 16 x 16 x 64 | 8 x 8 x 64
    c = BatchNormalization()(c)
    c = Conv2D(64, kernel_size=3, padding="same", activation='relu',
               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=constants.seed_value), kernel_regularizer=regularizers.l2(weight_decay))(c) # 16 x 16 x 64 | 8 x 8 x 64
    c = BatchNormalization()(c)
    c = MaxPool2D(pool_size=(2,2), strides=(1,1), padding="same")(c) # 16 x 16 x 64 | 8 x 8 x 64
    c = Dropout(0.25)(c)
    ##
    c = Flatten()(c)
    c = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(c)
    c = BatchNormalization()(c)
    c = Dropout(0.5)(c)
    classification = Dense(constants.n_labels, activation='softmax', name='classification')(c)

    return encoded_input, classification

class EarlyStopping_ae(Callback):
    """ Stop training when the loss gets lower than val_loss.

        Arguments:
            patience: Number of epochs to wait after condition has been hit.
            After this number of no reversal, training stops.
            It starts working after 10% of epochs have taken place.
    """

    def __init__(self):
        super(EarlyStopping_ae, self).__init__()
        self.patience = patience
        self.prev_val_loss = float('inf')
        self.prev_val_rmse = float('inf')

        # best_weights to store the weights at which the loss crossing occurs.
        self.best_weights = None
        self.start = min(epochs // 20, 3)
        self.wait = 0

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited since loss crossed val_loss.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        rmse = logs.get('root_mean_squared_error')
        val_rmse = logs.get('val_root_mean_squared_error')

        if epoch < self.start:
            self.best_weights = self.model.get_weights()
        elif (loss < val_loss) or (rmse < val_rmse):
            self.wait += 1
        elif (val_rmse < self.prev_val_rmse):
            self.wait = 0
            self.prev_val_rmse = val_rmse
            self.best_weights = self.model.get_weights()            
        elif (val_loss < self.prev_val_loss):
            self.wait = 0
            self.prev_val_loss = val_loss
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
        print(f'Epochs waiting: {self.wait}')
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print("Restoring model weights from the end of the best epoch.")
            self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


def train_network_ae(prefix, es):
    histories = []

    for fold in range(constants.n_folds):
        training_data, _ =  dataset.get_training(fold)
        testing_data, _ = dataset.get_testing(fold)
        #
        truly_training = int(len(training_data)*truly_training_percentage)
        #
        validation_data = training_data[truly_training:]
        #
        training_data = training_data[:truly_training]
        #
        rmse = tf.keras.metrics.RootMeanSquaredError()
        optimizer = tf.keras.optimizers.RMSprop()
        #
        input_enc, encoded = get_encoder()
        encoder = Model(input_enc, encoded, name='encoder')
        encoder.compile(optimizer=optimizer)
        encoder.summary()
        
        input_dec, decoded = get_decoder()
        decoder = Model(input_dec, decoded, name='decoder')
        decoder.compile(loss='mean_squared_error', optimizer=optimizer, metrics=rmse)
        decoder.summary()
        #
        input_img = Input(shape=(img_columns, img_rows, 3))

        encoded = encoder(input_img)
        decoded = decoder(encoded)

        autoencoder = Model(inputs=input_img, outputs=decoded, name='autoencoder')
        autoencoder.compile(loss='mean_squared_error',
                    optimizer=optimizer,
                    metrics=rmse)
        autoencoder.summary()

        history = autoencoder.fit(training_data,
                training_data,
                batch_size=batch_size_ae,
                epochs=epochs,
                validation_data= (validation_data,
                    validation_data),
                callbacks=[EarlyStopping_ae()],
                verbose=2)
        histories.append(history)
        print(f"\nAutoencoder evaluation on testing data")
        history = autoencoder.evaluate(testing_data, testing_data, return_dict=True)
        histories.append(history)
        encoder.save(constants.encoder_filename(prefix, es, fold))
        decoder.save(constants.decoder_filename(prefix, es, fold))
    return histories


class EarlyStopping_cla(Callback):
    """ Stop training when the loss gets lower than val_loss.

        Arguments:
            patience: Number of epochs to wait after condition has been hit.
            After this number of no reversal, training stops.
            It starts working after 10% of epochs have taken place.
    """

    def __init__(self):
        super(EarlyStopping_cla, self).__init__()
        self.patience = patience
        self.prev_val_loss = float('inf')
        self.prev_val_accuracy = 0.0

        # best_weights to store the weights at which the loss crossing occurs.
        self.best_weights = None
        self.start = min(epochs // 20, 3)
        self.wait = 0

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited since loss crossed val_loss.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        accuracy = logs.get('accuracy')
        val_accuracy = logs.get('val_accuracy')

        if epoch < self.start:
            self.best_weights = self.model.get_weights()
        elif (loss < val_loss) or (accuracy > val_accuracy):
            self.wait += 1
        elif (val_accuracy > self.prev_val_accuracy):
            self.wait = 0
            self.prev_val_accuracy = val_accuracy
            self.best_weights = self.model.get_weights()            
        elif (val_loss < self.prev_val_loss):
            self.wait = 0
            self.prev_val_loss = val_loss
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
        print(f'Epochs waiting: {self.wait}')
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print("Restoring model weights from the end of the best epoch.")
            self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


def train_network_cla(prefix, es):
    confusion_matrix = np.zeros((constants.n_labels, constants.n_labels))
    histories = []

    for fold in range(constants.n_folds):
        training_data, training_labels =  dataset.get_training(fold)
        testing_data, testing_labels = dataset.get_testing(fold)
        #
        truly_training = int(len(training_labels)*truly_training_percentage)
        #
        validation_data = training_data[truly_training:]
        validation_labels = training_labels[truly_training:]
        #
        training_data = training_data[:truly_training]
        training_labels = training_labels[:truly_training]
        #
        training_labels = to_categorical(training_labels)
        validation_labels = to_categorical(validation_labels)
        testing_labels = to_categorical(testing_labels)
        #
        # ==> Data augmentation
        data_augm = np.array(training_data, copy=True)
        lbl_augm = np.array(training_labels, copy=True)

        data_augm = tf.image.flip_left_right(data_augm)
        data_augm = tf. keras.layers.RandomZoom(height_factor=(-0.05,0.05), width_factor=(-0.05,0.05), \
                                                fill_mode='constant')(data_augm)
        len_data_augm = len(data_augm)
        data_augm_1 = data_augm[:len_data_augm//2]
        data_augm_1 = tfa.image.rotate(data_augm_1, (2*math.pi)/180)
        data_augm_2 = data_augm[len_data_augm//2:]
        data_augm_2 = tfa.image.rotate(data_augm_2, (-2*math.pi)/180)

        training_data = np.concatenate((training_data, data_augm_1, data_augm_2), axis=0)
        training_labels = np.concatenate((training_labels, lbl_augm), axis=0)
        training_data, training_labels = shuffle(training_data, training_labels)
        # <==
        # ==> Load encoder and predict
        encoder = tf.keras.models.load_model(constants.encoder_filename(prefix, es, fold))
        encoder.summary()
        encoded_training_data = encoder.predict(training_data)
        encoded_validation_data = encoder.predict(validation_data)
        #
        input_cla, classified = get_classifier()
        classifier = Model(input_cla, classified, name='classifier')
        classifier.compile(
            loss = 'categorical_crossentropy', optimizer = 'adam',
            metrics = 'accuracy')
        classifier.summary()
        #
        input_img = Input(shape=(img_columns, img_rows, 3))
        encoded = encoder(input_img)
        classified = classifier(encoded)
        full_classifier = Model(inputs=input_img, outputs=classified, name='full_classifier')
        full_classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = 'accuracy')
        full_classifier.summary()
        #
        history = classifier.fit(encoded_training_data,
                training_labels,
                batch_size=batch_size_cla,
                epochs=epochs,
                validation_data= (encoded_validation_data, validation_labels),
                callbacks=[EarlyStopping_cla()],
                verbose=2)
        #
        histories.append(history)
        print(f"\nFull classifier evaluation on testing data")
        history = full_classifier.evaluate(testing_data, testing_labels, return_dict=True)
        histories.append(history)
        predicted_labels = np.argmax(full_classifier.predict(testing_data), axis=1)
        confusion_matrix += tf.math.confusion_matrix(np.argmax(testing_labels, axis=1), predicted_labels, num_classes=constants.n_labels)
        classifier.save(constants.classifier_filename(prefix, es, fold))
        prediction_prefix = constants.classification_name(es)
        prediction_filename = constants.data_filename(prediction_prefix, es, fold)
        np.save(prediction_filename, predicted_labels)
    confusion_matrix = confusion_matrix.numpy()
    totals = confusion_matrix.sum(axis=1).reshape(-1,1)
    return histories, confusion_matrix/totals

def obtain_features(model_prefix, features_prefix, labels_prefix, data_prefix, es):
    """ Generate features for sound segments, corresponding to phonemes.
    
    Uses the previously trained neural networks for generating the features.
    """
    for fold in range(constants.n_folds):
        # Load de encoder
        filename = constants.encoder_filename(model_prefix, es, fold)
        model = tf.keras.models.load_model(filename)
        model.summary()

        noised_data, noised_labels = dataset.get_testing(fold, noised = True)
        partial_data, partial_labels = dataset.get_testing(fold, partial = True)
        training_data, training_labels = dataset.get_training(fold)
        filling_data, filling_labels = dataset.get_filling(fold)
        testing_data, testing_labels = dataset.get_testing(fold)

        settings = [
            (training_data, training_labels, constants.training_suffix),
            (filling_data, filling_labels, constants.filling_suffix),
            (testing_data, testing_labels, constants.testing_suffix),
            (noised_data, noised_labels, constants.noised_suffix),
            (partial_data, partial_labels, constants.partial_suffix)
        ]
        for s in settings:
            data = s[0]
            labels = s[1]
            suffix = s[2]
            features_filename = \
                constants.data_filename(features_prefix + suffix, es, fold)
            labels_filename = \
                constants.data_filename(labels_prefix + suffix, es, fold)
            features = model.predict(data)
            np.save(features_filename, features)
            np.save(labels_filename, labels)
