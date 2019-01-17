import numpy as np
import matplotlib.pyplot as plt
import sys



def showFigure(xs, ys, legends, y_title, title):
    for i in range(len(xs)):
        plt.plot(xs[i], ys[i])
    plt.ylabel(y_title.title(), fontsize=22)
    plt.xlabel('Epoch', fontsize=22)
    plt.title(title.title(), fontsize=26)
    plt.tick_params(labelsize=20)
    plt.legend(legends, prop={'size': 20})

    plt.show()

def showLossCompareFigure(epochs, losses, legends, title):
    showFigure(epochs, losses, legends, 'loss', title)

def showAccuracyFigure(epochs, accuracies, legends, title):
    showFigure(epochs, accuracies * 100, legends, 'accuracy(%)', title)



if __name__ == '__main__':
    if len(sys.argv) > 1:
        q_no = sys.argv[1]
    epoch = []
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    i = -1
    with open('mymodel_420test.txt') as f:
        for line in f:
            if line.startswith('Epoch'):
                temp = line.split(' ')[1].split('/')
                e = int(temp[0])
                if e == 0:
                    i += 1
                    epoch.append([])
                    train_loss.append([])
                    test_loss.append([])
                    train_accuracy.append([])
                    test_accuracy.append([])

                epoch[i].append(int(temp[0]))

            if line.startswith('train Loss'):
                temp = line.split(' ')
                train_loss[i].append(float(temp[2]))
                train_accuracy[i].append(float(temp[4]))

            if line.startswith('test Loss'):
                temp = line.split(' ')
                test_loss[i].append(float(temp[2]))
                test_accuracy[i].append(float(temp[4]))

    train_loss = np.array(train_loss)
    train_accuracy = np.array(train_accuracy)
    test_loss = np.array(test_loss)
    test_accuracy = np.array(test_accuracy)

    # showLossCompareFigure(epoch, test_loss, ['x0.9','nx'], 'Loss Comparison')
    showAccuracyFigure(epoch, test_accuracy, ['no text information','with text information'], 'Accuracy Comparison')
    # showLossCompareFigure(epoch, train_loss, ['x0.9','nx'], 'Loss Comparison')
    # showAccuracyFigure(epoch, train_accuracy, ['x0.9','nx'], 'Accuracy comparison')



