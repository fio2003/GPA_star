#!/usr/bin/env python3
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import figaspect


def main():
    filenames_found = [f.split("/")[-1] for f in os.listdir('./') if '.npy' in f]
    fig_num = 0
    for file in filenames_found:
        cur_arr = np.load(file)
        cur_arr = cur_arr.swapaxes(0, 1)
        new_name = file.split('.')[0]
        ax_prop = {"min_lim_x": min(cur_arr[0]), "max_lim_x": max(cur_arr[0]) + max(cur_arr[0]) / 80, "min_lim_y": min(cur_arr[1]), "max_lim_y": max(cur_arr[1]) - max(cur_arr[1]) / 80,
                   "min_ax_x": 0, "max_ax_x": max(cur_arr[0]) + max(cur_arr[0]) / 80, "min_ax_y": min(cur_arr[1]) + min(cur_arr[1]) / 80, "max_ax_y": max(cur_arr[1]) - max(cur_arr[1]) / 80,
                   "ax_step_x": (max(cur_arr[0]) - 0) / 16,
                   "ax_step_y": (max(cur_arr[1]) - min(cur_arr[1])) / 20}
        extra_line = [{"ax_type": 'ver', "val": 0, "name": "simulation origin", "col": "darkmagenta"}]
        fig_num = single_plot(fig_num, ax_prop, [cur_arr[0]], [cur_arr[1]], ['LJ interaction value'], '-', 1.0, True, True, False, 'Time, ps', 'LJ-SR, kJ/mol', 'Lennard-Jones Short Range Protein-Protein Interaction', new_name, extra_line=extra_line)
        plt.close('all')


def single_plot(fig_num: int, ax_prop: dict, arr_A: list, arr_B: list, filenames_db: list, marker: str, mark_size: float,
                bsf: bool, rev: bool, shrink: bool, xlab: str, ylab: str,
                title: str, filename: str, extra_line: list = None, mdpi: int = 400) -> int:
    """

    Args:
        :param int fig_num:
        :param dict ax_prop:
        :param list arr_A:
        :param list arr_B:
        :param list filenames_db:
        :param str marker:
        :param float mark_size:
        :param bool bsf:
        :param bool rev:
        :param bool shrink:
        :param str xlab:
        :param str ylab:
        :param str title:
        :param str filename:
        :param list extra_line:
        :param int mdpi:

    Returns:
        :return: last figure number.
        :rtype: int
    """
    fig_num += 1

    w, h = figaspect(0.5)
    fig = plt.figure(fig_num, figsize=(w, h))

    ax = fig.gca()
    plt.xlim(ax_prop["min_lim_x"], ax_prop["max_lim_x"])
    plt.ylim(ax_prop["min_lim_y"], ax_prop["max_lim_y"])

    major_xticks = np.arange(ax_prop["min_ax_x"], ax_prop["max_ax_x"], ax_prop["ax_step_x"])
    major_yticks = np.arange(ax_prop["min_ax_y"], ax_prop["max_ax_y"], ax_prop["ax_step_y"])

    if major_xticks is not None:
        ax.set_xticks(major_xticks)
    if major_yticks is not None:
        ax.set_yticks(major_yticks)
    # if minor_xticks is not None:
    #     ax.set_xticks(minor_xticks, minor=True)
    # if minor_yticks is not None:
    #     ax.set_yticks(minor_yticks, minor=True)

    plt.grid(which='both')
    plt.xticks(rotation=30)
    plt.subplots_adjust(top=0.95, bottom=0.14, left=0.09, right=0.98)

    lines_b = []
    for i, bsf_trav_to_goal in enumerate(arr_A):
        if not shrink:  # use provided array arr_B
            if rev:
                line_b, = plt.plot(arr_A[i], arr_B[i], marker, markersize=mark_size)
            else:
                line_b, = plt.plot(arr_B[i], arr_A[i], marker, markersize=mark_size)
        else:  # generate array from 0 to len(arr_A)
            if rev:
                if bsf:
                    line_b, = plt.plot(arr_A[i], range(len(arr_A[i])), marker, markersize=mark_size)
                else:
                    line_b, = plt.plot(arr_A[i], arr_B[i], marker, markersize=mark_size)
            else:
                line_b, = plt.plot(range(len(arr_A[i])), arr_A[i], marker, markersize=mark_size)
        lines_b.append(line_b)

    if extra_line is not None:
        for el in extra_line:
            if el["ax_type"] == 'ver':
                straight_line = plt.axvline(x=el["val"], color=el["col"], linestyle='-')  #
            elif el["ax_type"] == 'hor':
                straight_line = plt.axhline(y=el["val"], color=el["col"], linestyle='-')
            else:
                raise Exception('Wrong ax type')
            lines_b.append(straight_line)
            filenames_db.append(el["name"])
        # if el["ax_type"] == 'ver':
        #     if not rev:
        #         ax.annotate('folding direction', xytext=(ax_prop["min_ax_x"] + 1 * ax_prop["ax_step_x"], ax_prop["max_lim_y"] - 1 * ax_prop["ax_step_y"]), xy=(ax_prop["min_ax_x"] + 5 * ax_prop["ax_step_x"], ax_prop["max_lim_y"] - 1 * ax_prop["ax_step_y"]), arrowprops={'arrowstyle': '->', 'lw': 1.5, 'color': 'mediumblue'}, va='center')  # -->
        #     else:
        #         ax.annotate('folding direction', xytext=(ax_prop["max_ax_x"] - 1 * ax_prop["ax_step_x"], ax_prop["max_lim_y"] - 1 * ax_prop["ax_step_y"]), xy=(ax_prop["max_ax_x"] - 5 * ax_prop["ax_step_x"], ax_prop["max_lim_y"] - 1 * ax_prop["ax_step_y"]), arrowprops={'arrowstyle': '->', 'lw': 1.5, 'color': 'mediumblue'}, va='center')  # -->
        # else:
        #     if not rev:
        #         ax.annotate('folding direction', xytext=(ax_prop["min_ax_x"] + 1 * ax_prop["ax_step_x"], ax_prop["max_lim_y"] - 1 * ax_prop["ax_step_y"]), xy=(ax_prop["min_ax_x"] + 1 * ax_prop["ax_step_x"], ax_prop["max_lim_y"] - 4 * ax_prop["ax_step_y"]), arrowprops={'arrowstyle': '->', 'lw': 1.5, 'color': 'mediumblue'}, ha='center')  # <--
        #     else:
        #         pass # does not exist
                # ax.annotate('folding direction', xytext=(ax_prop["min_ax_x"] + 1 * ax_prop["ax_step_x"], ax_prop["max_lim_y"] - 1 * ax_prop["ax_step_y"]), xy=(ax_prop["min_ax_x"] + 1 * ax_prop["ax_step_x"], ax_prop["max_lim_y"] - 4 * ax_prop["ax_step_y"]), arrowprops={'arrowstyle': '->', 'lw': 1.5, 'color': 'mediumblue'}, ha='center')  # -->

    ax.legend(lines_b, filenames_db)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    try:
        plt.savefig(filename, dpi=mdpi)
    except:
        plt.show()
    plt.close('all')
    return fig_num


if __name__ == '__main__':
    main()
