import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores, record):

    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()

    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    plt.plot(scores)
    plt.hlines(y=record, xmax=len(scores), xmin=0, colors='aqua')

    plt.ylim(ymin=0)

    plt.plot(mean_scores)

    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(round(mean_scores[-1], 2)))
    plt.text(len(scores) - 1, record, str(record))

    plt.show(block=False)
    plt.pause(.1)