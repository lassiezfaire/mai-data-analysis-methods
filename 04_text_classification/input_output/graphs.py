import matplotlib.pyplot as plt


def plot_acc_loss(history, metrics: str, test_metrics: float, model_name):
    plt.figure(figsize=(7, 3))

    plt.subplot(1, 2, 1) if metrics == 'accuracy' else plt.subplot(1, 2, 2)
    plt.plot(history.history[metrics], label=metrics)
    plt.plot(history.history['val_' + metrics], label='val_' + metrics)
    plt.axhline(y=test_metrics, color='r', linestyle='--', label='test_' + metrics)
    plt.title(f'{model_name} - {metrics}')
    plt.xlabel('epoch')
    plt.ylabel(metrics)
    plt.legend()