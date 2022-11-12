import matplotlib.pyplot as plt

def display_hist(hist):
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']

    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    plt.figure(figsize=(10, 8))

    plt.subplot(1, 2, 1)
    plt.title('Training vs. Validation Accuracy')
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.xlabel('No. epoch')

    plt.subplot(1, 2, 2)
    plt.title('Training vs. Validation Loss')
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='lower left')
    plt.ylabel('Loss')
    plt.xlabel('No. epoch')

    plt.show()