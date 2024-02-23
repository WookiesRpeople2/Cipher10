import tensorflow as tf


# HyperParams
num_epochs = 20
model_save = "./tensorflow/trained_model.tnf"
# -------


def nuraleNet():

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(
        16, kernel_size=3, activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(
        32, kernel_size=3, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(10))

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True), metrics=['accuracy'])

    return model


if __name__ == "__main__":
    dataset = tf.keras.datasets.cifar10

    (X_train, y_train), (_, _) = dataset.load_data()

    # normalize the data to be from 0 - 1
    X_train = tf.keras.utils.normalize(X_train, axis=1)

    model = nuraleNet()

    model.fit(X_train, y_train, epochs=num_epochs)

    model.save(model_save)
