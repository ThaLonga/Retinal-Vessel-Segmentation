import matplotlib.pyplot as plt
from  keras.callbacks import Callback


def plot_images_with_masks(original, segmentation_mask, predicted_mask):
    """
    Plots the original image, segmentation mask, and predicted mask with a colorbar for the predicted mask.

    Parameters:
        original (ndarray): The original image.
        segmentation_mask (ndarray): The segmentation mask.
        predicted_mask (ndarray): The predicted mask.
    """

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Adjust size as needed

    # Show original image
    ax = axes[0]
    ax.imshow(original, cmap='gray')
    ax.set_title('Original Image')
    ax.axis('off')  # Hide axes ticks

    # Show segmentation mask
    ax = axes[1]
    im_seg = ax.imshow(segmentation_mask, cmap='plasma')
    ax.set_title('Manual Segmentation')
    ax.axis('off')

    # Show predicted mask
    ax = axes[2]
    im_pred = ax.imshow(predicted_mask, cmap='plasma')
    ax.set_title('Predicted Segmentation')
    ax.axis('off')

    # Colorbar for the predicted mask
    fig.colorbar(im_pred, ax=axes[2], fraction=0.046, pad=0.04)  # adjust fraction and pad to fit layout

    plt.tight_layout()
    plt.show()


class PredictionPlotter(Callback):
    def __init__(self, test_data, interval=10):
        super().__init__()
        self.test_data = test_data  # test_data should be a tuple (input_data, true_data)
        self.interval = interval    # Interval of epochs to plot predictions

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:  # +1 to count from 1
            predictions = self.model.predict(self.test_data[0])
            self.plot_predictions(predictions, epoch)

    def plot_predictions(self, predictions, epoch):
        plt.figure(figsize=(12, 6))
        for i, (img, pred) in enumerate(zip(self.test_data[1], predictions)):
            if i >= 1:  # Limit the number of images to plot
                break
            plt.subplot(2, 1, i + 1)
            plt.imshow(img.squeeze(), cmap='plasma')
            plt.title("True")
            plt.axis('off')

            plt.subplot(2, 1, 1 + i + 1)
            plt.imshow(pred.squeeze(), cmap='plasma')
            plt.title("Pred")
            plt.axis('off')

        plt.suptitle(f'Epoch: {epoch + 1}')
        plt.show()
