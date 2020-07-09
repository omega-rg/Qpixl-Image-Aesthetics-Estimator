import os

# tensorflow libraries
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# base model
from utils.nasnet import NASNetMobile

# image data generators
from utils.data_loader import train_generator,val_generator

# earth mover loss
def earth_mover_loss(y_true, y_pred):
    cdf_ytrue = K.cumsum(y_true, axis=-1)
    cdf_ypred = K.cumsum(y_pred, axis=-1)
    samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return K.mean(samplewise_emd)

# image dimension
image_size = 224

# Using NASNetMobile as the base model (Available by default with Keras as well)
base_model = NASNetMobile(input_shape=(image_size, image_size, 3), weights="imagenet", include_top=False, pooling='avg')
for layer in base_model.layers:
    layer.trainable = False

# Adding layers on top of the base model
x = Dropout(0.75)(base_model.output)
x = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)
# model.summary()

#Adam Optimizer
optimizer = Adam(learning_rate=1e-4)

#Compiling the model
model.compile(optimizer=optimizer, loss=earth_mover_loss)

# load weights from trained model if it exists
if os.path.exists('weights/nasnet_weights.h5'):
    model.load_weights('weights/nasnet_weights.h5')

# Model Checkpoint
checkpoint = ModelCheckpoint(filepath='weights/nasnet_weights.h5', monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True,
                             mode='min')
callbacks = [checkpoint]

# Batch Size and Epochs
batchsize = 100
epochs = 20

# Fitting the model
model.fit(train_generator(batchsize=batchsize),
                    steps_per_epoch=(250000. // batchsize),
                    epochs=epochs, verbose=1, callbacks=callbacks,
                    validation_data=val_generator(batchsize=batchsize),
                    validation_steps=(5000. // batchsize))
