import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# HyperParams
model_save = "./tensorflow/trained_model.tnf"
# -------


def drawplt(model, x, num_plots=30):
    preds = np.argmax(model.predict(x), axis=1)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    fig, axes = plt.subplots(num_plots//10, 10, figsize=(20, 15))
    fig.suptitle("Predictions")
    for i, ax in enumerate(axes.flat):
        ax.imshow(x[i])
        ax.set_title(f'Prediction: {classes[preds[i]]}')
        ax.axis('off')

    plt.show()


if __name__ == "__main__":
    dataset = tf.keras.datasets.cifar10

    (_, _), (X_test, _) = dataset.load_data()
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    model = tf.keras.models.load_model("./tensorflow/trained_model.tnf")

    drawplt(model, X_test, 40)
