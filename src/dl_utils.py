import gc
import json
import urllib.request
import zipfile
from glob import glob, iglob
from os.path import basename, exists
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.callbacks import History
from keras.layers import Input, InputLayer
from keras.models import Sequential
from keras.preprocessing import image
from keras.saving import load_model
import tensorflow as tf


def clean_memory():
    gc.collect()
    K.clear_session()


def preprocess_images(
    path, classes=None, preprocessor=None, ext=".jpg", **load_img_kwargs
):
    if classes is None:
        classes = ["."]
    X, y = [], []
    for classidx, classname in enumerate(classes):
        files = list(iglob(f"{path}/{classname}/*{ext}"))
        for img_idx, img_path in enumerate(files):
            print(f"{path}/{classname}: {img_idx} / {len(files)}", end="\r")
            img = image.load_img(img_path, **load_img_kwargs)
            x = image.img_to_array(img)
            img.close()
            if preprocessor:
                x = preprocessor(x)
            X.append(x)
            y.append(classidx)
        print()
    return np.array(X), np.array(y)


def download_image(url, convert_to_array=True, **load_img_kwargs):
    dest = download(url, overwrite=True)
    img = image.load_img(dest, **load_img_kwargs)
    return image.img_to_array(img) if convert_to_array else img


def download(url, dest_filename=None, overwrite=False):
    if not dest_filename:
        dest_filename = basename(url)
    if not overwrite and exists(dest_filename):
        print(
            f"'{dest_filename}' already exists, not overwriting (set overwrite=True to override)"
        )
        return
    print(f"Downloading to '{dest_filename}'...")
    urllib.request.urlretrieve(url, dest_filename)
    return dest_filename


def unzip(source_filename, dest_dir="."):
    print(f"Extracting '{source_filename}' to '{dest_dir}'...")
    with zipfile.ZipFile(source_filename) as zf:
        zf.extractall(dest_dir)


def clone_sequential_model(model, input_tensor=None):
    def _clone_layer(layer):
        return layer.__class__.from_config(layer.get_config())
    clone_function = _clone_layer
    new_layers = [clone_function(layer) for layer in model.layers]
    if isinstance(model._layers[0], InputLayer):
        ref_input_layer = model._layers[0]
        input_name = ref_input_layer.name
        input_batch_shape = ref_input_layer.batch_shape
        input_dtype = ref_input_layer._dtype
    else:
        input_name = None
        input_dtype = None
        input_batch_shape = None
    if input_tensor is not None:
        inputs = Input(tensor=input_tensor, name=input_name, shape=input_tensor.shape)
        new_layers = [inputs] + new_layers
    else:
        if input_batch_shape is not None:
            inputs = Input(
                tensor=[input_tensor],
                batch_shape=input_batch_shape,
                dtype=input_dtype,
                name=input_name,
            )
            new_layers = [inputs] + new_layers
    return Sequential(new_layers, name=model.name, trainable=model.trainable)


def get_model_weights(model):
    return {layer.name: layer.get_weights() for layer in model.layers}


def clone_and_set_weights(model, weights, input_tensor):
    new_model = clone_sequential_model(model, input_tensor=input_tensor)
    for layer in new_model.layers:
        layer.set_weights(weights[layer.name])
    return new_model


def fit_and_save(model, model_path=None, *fit_args, **fit_kwargs):
    if "learning_rate" in fit_kwargs:
        model.optimizer.lr = fit_kwargs["learning_rate"]
        del fit_kwargs["learning_rate"]
    history = model.fit(*fit_args, **fit_kwargs)
    if model_path:
        model.save(model_path)
    return history


def load_or_build(
    callable,
    model_path,
    *fit_and_save_args,
    **fit_and_save_kwargs,
):
    also_fit_if_not_loaded = True
    if "also_fit_if_not_loaded" in fit_and_save_kwargs:
        also_fit_if_not_loaded = fit_and_save_kwargs["also_fit_if_not_loaded"]
        del fit_and_save_kwargs["also_fit_if_not_loaded"]
    try:
        return load_model(model_path), True
    except Exception:
        pass
    model = callable()
    if also_fit_if_not_loaded:
        fit_and_save(model, model_path, *fit_and_save_args, **fit_and_save_kwargs)
    return model, False


def plot_history(history):
    history_metrics = list(history.history.keys())
    num_plots = len([name for name in history_metrics if not name.startswith("val_")])
    plt_index = 1

    plt.figure(figsize=(10, 3))
    plt.subplot(1, num_plots, plt_index)
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.plot(history.epoch, np.array(history.history["loss"]), label="Train Loss")
    if "val_loss" in history_metrics:
        plt.plot(
            history.epoch,
            np.array(history.history["val_loss"]),
            label="Validation Loss",
        )
    plt.legend()
    plt_index += 1

    for name in history_metrics:
        if name == "loss" or name.startswith("val_"):
            continue
        plt.subplot(1, num_plots, plt_index)
        plt.xlabel("Epoch")
        plt.ylabel(name)
        plt.plot(history.epoch, np.array(history.history[name]), label=f"Train {name}")
        if f"val_{name}" in history_metrics:
            plt.plot(
                history.epoch,
                np.array(history.history[f"val_{name}"]),
                label=f"Validation {name}",
            )
        plt.legend()
        plt_index += 1

    plt.show()


def merge_history(file_pattern_or_list):
    merged_history = {}
    if isinstance(file_pattern_or_list, str):
        file_names = sorted(glob(file_pattern_or_list))
        epochs = len(file_names)
        for file_name in file_names:
            history = json.load(open(file_name, "r"))
            for metric, vals in history.items():
                if metric not in merged_history:
                    merged_history[metric] = []
                merged_history[metric].extend(vals)
    else:
        epochs = len(file_pattern_or_list)
        for history in file_pattern_or_list:
            for metric, vals in history.items():
                if metric not in merged_history:
                    merged_history[metric] = []
                merged_history[metric].extend(vals)
    history = History()
    history.history = merged_history
    history.epoch = list(range(1, epochs + 1))
    return history


class KerasOptimizer(object):
    def __init__(self, losses):
        self.losses = losses

    def optimize(self, x, iterations=30, learning_rate=10.0, **kwargs):
        for _ in range(iterations):
            x, loss = self.sgd_step(x, learning_rate, **kwargs)
        return x, loss

    @tf.function
    def sgd_step(self, x, learning_rate, **kwargs):
        loss, grads = self.loss_and_grads(x, **kwargs)
        grads = tf.math.l2_normalize(grads)
        x -= learning_rate * grads
        return x, loss
    
    @tf.function
    def loss_and_grads(self, x, **kwargs):
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss = 0
            for loss_func in self.losses:
                loss += loss_func(x, **kwargs)
        grads = tape.gradient(loss, x)
        return loss, grads
