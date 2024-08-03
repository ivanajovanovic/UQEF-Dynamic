import pathlib
from plotly.offline import iplot, plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from uqef_dynamic.utils import colors
from uqef_dynamic.models.time_dependent_baseclass.time_dependent_statistics import TimeDependentStatistics

class simpleOscilatorStatistics(TimeDependentStatistics):
    def __init__(self, configurationObject, workingDir=None, *args, **kwargs):
        super().__init__(configurationObject, workingDir, *args, **kwargs)

    def prepare_for_plotting(self, timestep=-1, display=False,
                    fileName="", fileNameIdent="", directory="./",
                    fileNameIdentIsFullName=False, safe=True, **kwargs):
        pass

    def plotResults_single_qoi(self, single_qoi_column, dict_time_vs_qoi_stat=None, timestep=-1, display=False, fileName="",
                               fileNameIdent="", directory="./", fileNameIdentIsFullName=False, safe=True,
                               dict_what_to_plot=None, **kwargs):
        """
        This function plots the statistics of a single QoI.
        It is more relevant that the plotResults function, which plots all the QoIs.
        """
        if dict_time_vs_qoi_stat is None:
            dict_time_vs_qoi_stat = self.result_dict[single_qoi_column]

        if fileName == "":
            fileName = single_qoi_column

        single_fileName = self.generateFileName(fileName=fileName, fileNameIdent=".html",
                                                directory=directory, fileNameIdentIsFullName=fileNameIdentIsFullName)

        fig = self._plotStatisticsDict_plotly_single_qoi(single_qoi_column=single_qoi_column,
                                                         dict_time_vs_qoi_stat=dict_time_vs_qoi_stat,
                                                         filename=single_fileName, display=display,
                                                         dict_what_to_plot=dict_what_to_plot, **kwargs)
        if display:
            fig.show()
        print(f"[STAT INFO] plotResults for QoI-{single_qoi_column} function is done!")


    def plotResults(self, timestep=-1, display=False, fileName="", fileNameIdent="", directory="./",
                    fileNameIdentIsFullName=False, safe=True, dict_what_to_plot=None, **kwargs):
        """
        This function plots the statistics of a single, or multiple, QoI.
        Thake a look at the plotResults_single_qoi function for more details.
        """
        pass

    def _plotStatisticsDict_plotly_single_qoi(self, single_qoi_column, dict_time_vs_qoi_stat=None, 
                                              window_title='Forward UQ & SA', filename="sim-plotly.html", display=False,
                                              dict_what_to_plot=None, **kwargs):

        pdTimesteps = self.timesteps
        keyIter = list(pdTimesteps)

        window_title = window_title + f": QoI - {single_qoi_column}"

        if dict_time_vs_qoi_stat is None:
            dict_time_vs_qoi_stat = self.result_dict[single_qoi_column]

        if dict_what_to_plot is None:
            if self.dict_what_to_plot is not None:
                dict_what_to_plot = self.dict_what_to_plot
            else:
                dict_what_to_plot = {
                    "E_minus_std": False, "E_plus_std": False, "P10": False, "P90": False,
                    "StdDev": False, "Skew": False, "Kurt": False, "Sobol_m": False, "Sobol_m2": False, "Sobol_t": False
                }

        # self._check_if_Sobol_t_computed(keyIter[0], qoi_column=single_qoi_column)
        # self._check_if_Sobol_m_computed(keyIter[0], qoi_column=single_qoi_column)

        n_rows, starting_row = self._compute_number_of_rows_for_plotting(
            dict_what_to_plot, forcing=False, list_qoi_column_to_plot=[single_qoi_column,], result_dict=dict_time_vs_qoi_stat)

        fig = make_subplots(rows=n_rows, cols=1,
                            print_grid=True,
                            shared_xaxes=True,
                            vertical_spacing=0.1)

        dict_plot_rows = dict()

        if "E" in dict_time_vs_qoi_stat[keyIter[0]]:
            fig.add_trace(go.Scatter(x=pdTimesteps,
                                    y=[dict_time_vs_qoi_stat[key]["E"] for key in keyIter],
                                    name=f'E[{single_qoi_column}]',
                                    line_color='green', mode='lines'),
                        row=starting_row, col=1)

        if dict_what_to_plot.get("E_minus_std", False) and "StdDev" in dict_time_vs_qoi_stat[keyIter[0]] and "E" in dict_time_vs_qoi_stat[keyIter[0]]:
            fig.add_trace(go.Scatter(x=pdTimesteps,
                                     y=[(dict_time_vs_qoi_stat[key]["E"] \
                                         - dict_time_vs_qoi_stat[key]["StdDev"]) for key in keyIter],
                                     name='mean - std. dev', line_color='darkviolet', mode='lines'),
                          row=starting_row, col=1)
        if dict_what_to_plot.get("E_plus_std", False) and "StdDev" in dict_time_vs_qoi_stat[keyIter[0]] and "E" in dict_time_vs_qoi_stat[keyIter[0]]:
            fig.add_trace(go.Scatter(x=pdTimesteps,
                                     y=[(dict_time_vs_qoi_stat[key]["E"] + \
                                         dict_time_vs_qoi_stat[key]["StdDev"]) for key in keyIter],
                                     name='mean + std. dev', line_color='darkviolet', mode='lines', fill='tonexty'),
                          row=starting_row, col=1)
        if "P10" in dict_time_vs_qoi_stat[keyIter[0]] and dict_what_to_plot.get("P10", False):
            fig.add_trace(go.Scatter(x=pdTimesteps,
                                     y=[dict_time_vs_qoi_stat[key]["P10"] for key in keyIter],
                                     name='10th percentile', line_color='yellow', mode='lines'),
                          row=starting_row, col=1)
        if "P90" in dict_time_vs_qoi_stat[keyIter[0]] and dict_what_to_plot.get("P90", False):
            fig.add_trace(go.Scatter(x=pdTimesteps,
                                     y=[dict_time_vs_qoi_stat[key]["P90"] for key in keyIter],
                                     name='90th percentile', line_color='yellow', mode='lines', fill='tonexty'),
                          row=starting_row, col=1)
        dict_plot_rows["qoi"] = starting_row
        starting_row += 1

        # fig.add_trace(go.Scatter(x=pdTimesteps,
        #                          y=[dict_time_vs_qoi_stat[key]["StdDev"] for key in keyIter],
        #                          name='std. dev', line_color='darkviolet', mode='lines'),
        #               row=starting_row, col=1)
        # dict_plot_rows["StdDev"] = starting_row
        # starting_row += 1

        if "StdDev" in dict_time_vs_qoi_stat[keyIter[0]] and dict_what_to_plot.get("StdDev", False):
            fig.add_trace(go.Scatter(x=pdTimesteps,
                                     y=[dict_time_vs_qoi_stat[key]["StdDev"] for key in keyIter],
                                     name='std. dev', line_color='darkviolet', mode='lines'),
                          row=starting_row, col=1)
            dict_plot_rows["StdDev"] = starting_row
            starting_row += 1

        if "Skew" in dict_time_vs_qoi_stat[keyIter[0]] and dict_what_to_plot.get("Skew", False):
            fig.add_trace(go.Scatter(x=pdTimesteps,
                                     y=[dict_time_vs_qoi_stat[key]["Skew"] for key in keyIter],
                                     name='Skew', mode='lines', ),
                          row=starting_row, col=1)
            dict_plot_rows["Skew"] = starting_row
            starting_row += 1

        if "Kurt" in dict_time_vs_qoi_stat[keyIter[0]] and dict_what_to_plot.get("Kurt", False):
            fig.add_trace(go.Scatter(x=pdTimesteps,
                                     y=[dict_time_vs_qoi_stat[key]["Kurt"] for key in keyIter],
                                     name='Kurt', mode='lines', ),
                          row=starting_row, col=1)
            dict_plot_rows["Kurt"] = starting_row
            starting_row += 1

        if "Sobol_m" in dict_time_vs_qoi_stat[keyIter[0]] and dict_what_to_plot.get("Sobol_m", False):
            for i in range(len(self.labels)):
                name = self.labels[i] + "_" + single_qoi_column + "_S_m"
                fig.add_trace(go.Scatter(
                    x=pdTimesteps,
                    y=[dict_time_vs_qoi_stat[key]["Sobol_m"][i] for key in keyIter],
                    name=name, legendgroup=self.labels[i], line_color=colors.COLORS[i], mode='lines'),
                    row=starting_row, col=1)
            dict_plot_rows["Sobol_m"] = starting_row
            starting_row += 1

        if "Sobol_m2" in dict_time_vs_qoi_stat[keyIter[0]] and dict_what_to_plot.get("Sobol_m2", False):
            for i in range(len(self.labels)):
                name = self.labels[i] + "_" + single_qoi_column + "_S_m"
                fig.add_trace(go.Scatter(
                    x=pdTimesteps,
                    y=[dict_time_vs_qoi_stat[key]["Sobol_m2"][i] for key in keyIter],
                    name=name, legendgroup=self.labels[i], line_color=colors.COLORS[i], mode='lines'),
                    row=starting_row, col=1)
            dict_plot_rows["Sobol_m2"] = starting_row
            starting_row += 1

        if "Sobol_t" in dict_time_vs_qoi_stat[keyIter[0]] and dict_what_to_plot.get("Sobol_t", False):
            for i in range(len(self.labels)):
                name = self.labels[i] + "_" + single_qoi_column + "_S_t"
                fig.add_trace(go.Scatter(
                    x=pdTimesteps,
                    y=[dict_time_vs_qoi_stat[key]["Sobol_t"][i] for key in keyIter],
                    name=name, legendgroup=self.labels[i], line_color=colors.COLORS[i], mode='lines'),
                    row=starting_row, col=1)
            dict_plot_rows["Sobol_t"] = starting_row
            starting_row += 1

        plot_generalized_sobol_indices = False
        for key in dict_time_vs_qoi_stat[keyIter[-1]].keys():
            if key.startswith("generalized_sobol_total_index_"):
                plot_generalized_sobol_indices = True
                break
        if plot_generalized_sobol_indices:  
            for i in range(len(self.labels)):
                name = f"generalized_sobol_total_index_{self.labels[i]}"
                if self.compute_generalized_sobol_indices_over_time:
                    y = [dict_time_vs_qoi_stat[key][name] for key in keyIter]
                else:
                    y = [dict_time_vs_qoi_stat[keyIter[-1]][name]]*len(keyIter)
                fig.add_trace(go.Scatter(
                    x=pdTimesteps,
                    y=y,
                    name=name, legendgroup=self.labels[i], line_color=colors.COLORS[i], mode='lines'),
                    row=starting_row, col=1)
            dict_plot_rows["generalized_sobol_total_index"] = starting_row
            starting_row += 1

        fig.update_yaxes(title_text=single_qoi_column, showgrid=True, side='left',
                         row=dict_plot_rows["qoi"], col=1)
        # fig.update_yaxes(title_text=f"Std. Dev. [{single_qoi_column}]", side='left', showgrid=True,
        #                  row=starting_row+1, col=1)
        if "StdDev" in dict_time_vs_qoi_stat[keyIter[0]] and dict_what_to_plot.get("StdDev", False):
            fig.update_yaxes(title_text=f"StdDev", showgrid=True,
                             row=dict_plot_rows["StdDev"], col=1)
        if "Skew" in dict_time_vs_qoi_stat[keyIter[0]] and dict_what_to_plot.get("Skew", False):
            fig.update_yaxes(title_text=f"Skew", showgrid=True,
                             row=dict_plot_rows["Skew"], col=1)
        if "Kurt" in dict_time_vs_qoi_stat[keyIter[0]] and dict_what_to_plot.get("Kurt", False):
            fig.update_yaxes(title_text=f"Kurt", showgrid=True,
                             row=dict_plot_rows["Kurt"], col=1)
        if "Sobol_m" in dict_time_vs_qoi_stat[keyIter[0]] and dict_what_to_plot.get("Sobol_m", False):
            fig.update_yaxes(title_text=f"F. SI", showgrid=True, range=[0, 1],
                             row=dict_plot_rows["Sobol_m"], col=1)
        if "Sobol_m2" in dict_time_vs_qoi_stat[keyIter[0]] and dict_what_to_plot.get("Sobol_m2", False):
            fig.update_yaxes(title_text=f"S. SI", showgrid=True, range=[0, 1],
                             row=dict_plot_rows["Sobol_m2"], col=1)
        if "Sobol_t" in dict_time_vs_qoi_stat[keyIter[0]] and dict_what_to_plot.get("Sobol_t", False):
            fig.update_yaxes(title_text=f"T. SI", showgrid=True, range=[0, 1],
                             row=dict_plot_rows["Sobol_t"], col=1)
        if plot_generalized_sobol_indices and dict_plot_rows.get("generalized_sobol_total_index", False):
            fig.update_yaxes(title_text=f"Gener. T. SI", showgrid=True, range=[0, 1],
                             row=dict_plot_rows["generalized_sobol_total_index"], col=1)

        fig.update_layout(width=1000)
        fig.update_layout(title_text=window_title)
        # fig.update_layout(xaxis=dict(type="date"))

        print(f"[SIMPLE OSCILATOR STAT INFO] _plotStatisticsDict_plotly function for Qoi-{single_qoi_column} is almost over, just to save the plot!")

        # filename = pathlib.Path(filename)
        plot(fig, filename=filename, auto_open=display)
        return fig
