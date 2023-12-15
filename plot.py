import pandas as pd
import matplotlib.pyplot as plt


def main(file, device):
    data = pd.read_csv(file)
    plt.plot(data['accuracy'], label='Accuracy')
    plt.plot(data['loss'], label='Loss')
    plt.plot(data['val_accuracy'], label='Validation Accuracy')
    plt.plot(data['val_loss'], label='Validation Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title(f'Training Metrics {device}')
    plt.legend()
    plt.savefig(f'plots/metrics_{device}.png')
    plt.clf()
    
    epochs = range(1, len(data['time']) + 1)
    plt.bar(x=epochs, height=data['time'], label='Time')
    plt.xlabel('Epochs')
    plt.ylabel('Time')
    plt.title(f'Time {device}')
    plt.legend()
    plt.savefig(f'plots/time_{device}.png')
    plt.clf()
    
    
if __name__ == '__main__':
    main('data/history_cpu.csv', 'cpu')
    main('data/history_gpu.csv', 'gpu')
    
