"""
code by Gourav Jhanwar
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from skimage import measure
from portpy.photon.evaluation import Evaluation
from matplotlib.lines import Line2D
import os
from portpy.photon.utils import view_in_slicer_jupyter
from pathlib import Path
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from portpy.photon.plan import Plan
    from portpy.photon.influence_matrix import InfluenceMatrix
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class Visualization:
    dose_type = Literal["Absolute(Gy)", "Relative(%)"]
    volume_type = Literal["Absolute(cc)", "Relative(%)"]
    
    def plot_robust_dvh(my_plan: Plan, sol: dict = None, dose_1d_list: list = None, struct_names: List[str] = None,
                 dose_scale: dose_type = "Absolute(Gy)",
                 volume_scale: volume_type = "Relative(%)", plot_scenario=None, **options):
        """
        Create dvh plot for the selected structures

        :param my_plan: object of class Plan
        :param sol: optimal sol dictionary
        :param dose_1d: dose_1d in 1d voxels
        :param struct_names: structures to be included in dvh plot
        :param volume_scale: volume scale on y-axis. Default= Absolute(cc). e.g. volume_scale = "Absolute(cc)" or volume_scale = "Relative(%)"
        :param dose_scale: dose_1d scale on x axis. Default= Absolute(Gy). e.g. dose_scale = "Absolute(Gy)" or dose_scale = "Relative(%)"
        :keyword style (str): line style for dvh curve. default "solid". can be "dotted", "dash-dotted".
        :keyword width (int): width of line. Default 2
        :keyword colors(list): list of colors
        :keyword legend_font_size: Set legend_font_size. default 10
        :keyword figsize: Set figure size for the plot. Default figure size (12,8)
        :keyword create_fig: Create a new figure. Default True. If False, append to the previous figure
        :keyword title: Title for the figure
        :keyword filename: Name of the file to save the figure in current directory
        :keyword show: Show the figure. Default is True. If false, next plot can be append to it
        :keyword norm_flag: Use to normalize the plan. Default is False.
        :keyword norm_volume: Use to set normalization volume. default is 90 percentile.
        :return: dvh plot for the selected structures

        :Example:
        >>> Visualization.plot_dvh(my_plan, sol=sol, struct_names=['PTV', 'ESOPHAGUS'], dose_scale='Absolute(Gy)',volume_scale="Relative(%)", show=False, create_fig=True )
        """

        if not isinstance(dose_1d_list, list):
            dose_1d_list = [dose_1d_list]
        if len(dose_1d_list) == 0:
            raise ValueError("dose_list is empty")
        if sol is None:
            sol = dict()
            sol['inf_matrix'] = my_plan.inf_matrix  # create temporary solution

        if dose_1d_list is None:
            dose_1d_list = []
            if isinstance(sol, list):
                for s in sol:
                    if 'inf_matrix' not in s:
                        s['inf_matrix'] = my_plan.inf_matrix
                    dose_1d_list += [s['inf_matrix'].A @ (s['optimal_intensity'] * my_plan.get_num_of_fractions())]

        # getting options_fig:
        style = options['style'] if 'style' in options else 'solid'
        width = options['width'] if 'width' in options else None
        colors = options['colors'] if 'colors' in options else None
        legend_font_size = options['legend_font_size'] if 'legend_font_size' in options else 15
        figsize = options['figsize'] if 'figsize' in options else (12, 8)
        title = options['title'] if 'title' in options else None
        filename = options['filename'] if 'filename' in options else None
        show = options['show'] if 'show' in options else False
        # create_fig = options['create_fig'] if 'create_fig' in options else False
        show_criteria = options['show_criteria'] if 'show_criteria' in options else None
        ax = options['ax'] if 'ax' in options else None
        fontsize = options['fontsize'] if 'fontsize' in options else 12
        legend_loc = options["legend_loc"] if "legend_loc" in options else "upper right"
        # getting norm options
        norm_flag = options['norm_flag'] if 'norm_flag' in options else False
        norm_volume = options['norm_volume'] if 'norm_volume' in options else 90
        norm_struct = options['norm_struct'] if 'norm_struct' in options else 'PTV'

        # plt.rcParams['font.size'] = font_size
        # plt.rc('font', family='serif')
        if width is None:
            width = 3
        if colors is None:
            colors = Visualization.get_colors()
        if struct_names is None:
            # orgs = []
            struct_names = my_plan.structures.structures_dict['name']
        max_dose = 0.0
        max_vol = 0.0
        all_orgs = my_plan.structures.structures_dict['name']
        # orgs = [struct.upper for struct in orgs]
        pres = my_plan.get_prescription()
        legend = []

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        # if norm_flag:
        #     norm_factor = Evaluation.get_dose(sol, dose_1d=dose_1d, struct=norm_struct, volume_per=norm_volume) / pres
        #     dose_1d = dose_1d / norm_factor
        count = 0
        for i in range(np.size(all_orgs)):
            if all_orgs[i] not in struct_names:
                continue
            if my_plan.structures.get_fraction_of_vol_in_calc_box(all_orgs[i]) == 0:  # check if the structure is within calc box
                print('Skipping Structure {} as it is not within calculation box.'.format(all_orgs[i]))
                continue
            dose_sort_list = []
            for dose_1d in dose_1d_list:
                x, y = Evaluation.get_dvh(sol, struct=all_orgs[i], dose_1d=dose_1d)
                dose_sort_list.append(x)
            d_sort_mat = np.column_stack(dose_sort_list)
            # Compute min/max DVH curve taken across scenarios.
            d_min_mat = np.min(d_sort_mat, axis=1)
            d_max_mat = np.max(d_sort_mat, axis=1)

            if dose_scale == 'Absolute(Gy)':
                max_dose = np.maximum(max_dose, d_max_mat[-1])
                ax.set_xlabel('Dose (Gy)', fontsize=fontsize)
            elif dose_scale == 'Relative(%)':
                max_dose = np.maximum(max_dose, d_max_mat[-1])
                max_dose = max_dose/pres*100
                ax.set_xlabel('Dose ($\%$)', fontsize=fontsize)

            if volume_scale == 'Absolute(cc)':
                y = y * my_plan.structures.get_volume_cc(all_orgs[i]) / 100
                max_vol = np.maximum(max_vol, y[1] * 100)
                ax.set_ylabel('Volume (cc)', fontsize=fontsize)
            elif volume_scale == 'Relative(%)':
                max_vol = np.maximum(max_vol, y[0] * 100)
                ax.set_ylabel('Volume Fraction ($\%$)', fontsize=fontsize)
            # ax.plot(x, 100 * y, linestyle=style, linewidth=width, color=colors[count])

            # ax.plot(d_min_mat, 100 * y, linestyle='dotted', linewidth=width*0.5, color=colors[count])
            # ax.plot(d_max_mat, 100 * y, linestyle='dotted', linewidth=width*0.5, color=colors[count])
            ax.fill_betweenx(100 * y, d_min_mat, d_max_mat, alpha=0.25, color=colors[count])

            # Plot user-specified scenarios.
            if plot_scenario is not None:
                if plot_scenario == 'mean':
                    dose_mean = np.mean(d_sort_mat, axis=1)
                    ax.plot(dose_mean, 100 * y, linestyle=style, color=colors[count], linewidth=width, label=all_orgs[i])
                elif not isinstance(plot_scenario, list):
                    plot_scenario = [plot_scenario]

                    for n in range(len(plot_scenario)):
                        scene_num = plot_scenario[n]
                        if norm_flag:
                            norm_factor = Evaluation.get_dose(sol, dose_1d=dose_1d_list[scene_num], struct=norm_struct, volume_per=norm_volume) / pres
                            dose_sort_list[scene_num] = dose_sort_list[scene_num] / norm_factor
                            d_min_mat = d_min_mat / norm_factor
                            d_max_mat = d_max_mat / norm_factor
                        ax.plot(dose_sort_list[scene_num], 100 * y, linestyle=style, color=colors[count], linewidth=width, label=all_orgs[i])
            count = count + 1
            # legend.append(all_orgs[i])

        if show_criteria is not None:
            for s in range(len(show_criteria)):
                if 'dose_volume' in show_criteria[s]['type']:
                    x = show_criteria[s]['parameters']['dose_gy']
                    y = show_criteria[s]['constraints']['limit_volume_perc']
                    ax.plot(x, y, marker='x', color='red', markersize=20)
        # plt.xlabel('Dose (Gy)')
        # plt.ylabel('Volume Fraction (%)')
        current_xlim = ax.get_xlim()
        final_xmax = max(current_xlim[1], max_dose * 1.1)
        ax.set_xlim(0, final_xmax)
        ax.set_ylim(0, max_vol)
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique), prop={'size': legend_font_size}, loc=legend_loc)
        # ax.legend(legend, prop={'size': legend_font_size}, loc=legend_loc)
        ax.grid(visible=True, which='major', color='#666666', linestyle='-')

        # Show the minor grid lines with very faint and almost transparent grey lines
        # plt.minorticks_on()
        ax.minorticks_on()
        plt.grid(visible=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
        y = np.arange(0, 101)
        # if norm_flag:
        #     x = pres * np.ones_like(y)
        # else:
        if dose_scale == "Absolute(Gy)":
            x = pres * np.ones_like(y)
        else:
            x = 100 * np.ones_like(y)

        ax.plot(x, y, color='black')
        if title:
            ax.set_title(title)
        if show:
            plt.show()
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight", dpi=300)
        return ax
    
    @staticmethod
    def get_colors():
        """

        :return: return list of 19 colors
        """
        # colors = ['#4363d8', '#f58231', '#911eb4',
        #           '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
        #           '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
        #           '#000075', '#808080', '#ffffff', '#e6194b', '#3cb44b']
        colors = [
            "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e",
            "#8c564b", "#e377c2", "#7f7f7f", "#17becf", "#bcbd22",
            "#20b2aa", "#ff00ff", "#ffff00", "#87ceeb", "#006400",
            "#fa8072", "#e6e6fa", "#ffd700", "#8b0000", "#40e0d0"
        ]
        return colors