from matplotlib import pyplot

class DataPlotter:
    def __init__(self, plots, labels, figsize=None, plot_func="loglog"):
        self.plots = plots
        self.labels = labels
        self.plot_func = plot_func

        n_plots = len(plots)
        if figsize is None:
            figsize = [15, 8]
        elif isinstance(figsize, tuple):
            figsize = list(figsize)
        figsize[1] *= n_plots
        self.fig, axl = pyplot.subplots(n_plots, 1, figsize=figsize)
        self.axes = {plot[0]: axes for plot, axes in zip(plots, axl)}
        
    def plot_data(self, data, color):
        for plot in self.plots:
            y_field, x_field = plot
            y_data = data.get(plot[0])
            x_data = data.get(plot[1])
            if None in (x_data, y_data):
                continue
            my_plot = getattr(self.axes[y_field], self.plot_func)
            my_plot(x_data, y_data, color=color)
            
    def plot_axes(self):
        for plot in self.plots:
            y_field, x_field = plot
            axes = self.axes[y_field]
            axes.xaxis.set_label_text(self.labels[x_field])
            axes.yaxis.set_label_text(self.labels[y_field])
