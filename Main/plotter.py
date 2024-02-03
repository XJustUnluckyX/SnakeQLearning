import matplotlib.pyplot as plotter
from IPython import display

plotter.ion()


def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plotter.gcf())
    plotter.clf()
    plotter.title('Addestramento...')
    plotter.xlabel('Numero di partite')
    plotter.ylabel('Score')
    plotter.plot(scores)
    plotter.plot(mean_scores)
    plotter.ylim(ymin=0)
    plotter.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plotter.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plotter.show(block=False)
    plotter.pause(.1)
