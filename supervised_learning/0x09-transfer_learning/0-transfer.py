#!/usr/bin/env python3
"""
Trains a convolutional neural network to classify the CIFAR 10 dataset
"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    Pre-processes the data for the model
    """
    X_p = K.applications.densenet.preprocess_input(X)  # normalizar
    Y_p = K.utils.to_categorical(Y, 10)  # matrix one hot encode
    return X_p, Y_p


if __name__ == "__main__":
    """
    Trains a convolutional neural network to classify the CIFAR 10 dataset
    """
    kernel_norm = K.initializers.he_normal()
    input_tensor = K.Input(shape=(32, 32, 3))
    (Xt, Yt), (X, Y) = K.datasets.cifar10.load_data()
    X_p, Y_p = preprocess_data(Xt, Yt)
    Xv_p, Yv_p = preprocess_data(X, Y)
    # resize images to the image size upon which the network was pre-trained
    new_size = K.layers.Lambda(lambda image: tf.image.resize(
                               image, (224, 224)))(
                               input_tensor)
    base_model = K.applications.DenseNet121(include_top=False,
                                            weights='imagenet',
                                            input_tensor=new_size,
                                            input_shape=(224, 224, 3),
                                            pooling='max')
    # make the weights and biases of the base model non-trainable
    # by "freezing" each layer of the DenseNet201 network
    base_model.trainable = False
    # add more layers
    # take output from base_model without last layer
    output = base_model.output
    flat = K.layers.Flatten()(output)
    batch = K.layers.BatchNormalization()(flat)  # Normalize the flat output
    dense = K.layers.Dense(256, activation="relu",
                           kernel_initializer=kernel_norm)(batch)
    drop = K.layers.Dropout(0.4)(dense)
    batch1 = K.layers.BatchNormalization()(drop)  # avoid overfitting
    dense1 = K.layers.Dense(128, activation="relu",
                            kernel_initializer=kernel_norm)(batch1)
    drop1 = K.layers.Dropout(0.4)(dense1)
    batch2 = K.layers.BatchNormalization()(drop1)  # avoid overfitting
    dense2 = K.layers.Dense(64, activation="relu",
                            kernel_initializer=kernel_norm)(batch2)
    drop2 = K.layers.Dropout(0.4)(dense2)
    # output layer
    out = K.layers.Dense(10, activation="softmax")(drop2)
    callback = K.callbacks.ModelCheckpoint(filepath='cifar10.h5',
                                           monitor='val_acc',
                                           mode='max',
                                           save_best_only=True)
    model = K.models.Model(inputs=input_tensor, outputs=out)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=X_p, y=Y_p, validation_data=(Xv_p, Yv_p), batch_size=128,
              epochs=20, callbacks=[callback], verbose=1)
