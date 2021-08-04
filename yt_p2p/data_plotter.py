from matplotlib import pyplot

class DataPlotter:
    def __init__(self, plots, labels, figsize=None, plot_func="loglog"):
        self.plots = plots
        self.labels = labels
        self.plot_func = plot_func
        self.legends = []

        n_plots = len(plots)
        if figsize is None:
            figsize = [15, 8]
        elif isinstance(figsize, tuple):
            figsize = list(figsize)
        figsize[1] *= n_plots
        self.fig, axl = pyplot.subplots(n_plots, 1, figsize=figsize)
        self.axes = {plot[0]: axes for plot, axes in zip(plots, axl)}
        
    def plot_data(self, data, **pkwargs):
        for plot in self.plots:
            y_field, x_field = plot
            y_data = data.get(plot[0])
            x_data = data.get(plot[1])
            if x_data is None or y_data is None:
                continue
            self.plot_datum(y_field, x_data, y_data, **pkwargs)

    def plot_datum(self, pfield, x_data, y_data, **pkwargs):
        if "label" in pkwargs and pfield not in self.legends:
            self.legends.append(pfield)
        my_plot = getattr(self.axes[pfield], self.plot_func)
        my_plot(x_data, y_data, **pkwargs)

    def plot_axes(self):
        for plot in self.plots:
            y_field, x_field = plot
            axes = self.axes[y_field]
            axes.xaxis.set_label_text(self.labels[x_field])
            axes.yaxis.set_label_text(self.labels[y_field])
            if y_field in self.legends:
                axes.legend(loc='best')
