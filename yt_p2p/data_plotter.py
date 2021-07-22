from matplotlib import pyplot

class DataPlotter:
    def __init__(self, plots, labels):
        self.plots = plots
        self.labels = labels
        n_plots = len(plots)
        self.fig, axl = pyplot.subplots(n_plots, 1, figsize=(15, n_plots * 8))
        self.axes = {plot[0]: axes for plot, axes in zip(plots, axl)}
        
    def plot_data(self, data, color):
        for plot in self.plots:
            y_data = data[plot[0]]
            x_data = data[plot[1]]
            self.axes[plot[0]].loglog(x_data, y_data, color=color)
            
    def plot_axes(self):
        for plot in self.plots:
            y_field, x_field = plot
            axes = self.axes[y_field]
            axes.xaxis.set_label_text(self.labels[x_field])
            axes.yaxis.set_label_text(self.labels[y_field])
