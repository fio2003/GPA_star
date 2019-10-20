#!/usr/bin/env python3

import os
import sqlite3 as lite
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import figaspect
import multiprocessing as mp
import math


def main():
    """
    This function sets the task.
    Our task is to compare different runs by plotting plots.
    You specify DB names and proper legend entrees
    """
    batch_arr = list()
    ffs = ['amber', 'charm', 'gromos', 'opls']
    #########   TRP   ######################
    # for ff in ffs:
    #     filenames_db = ['results_{}_trp_300_fixed.sqlite3'.format(ff), 'results_{}_trp_300_2_fixed.sqlite3'.format(ff)]
    #     legend_names = ['TRP {}_1'.format(ff), 'TRP {}_2'.format(ff)]
    #     common_path = '../trp_{}_compar'.format(ff)
    #     batch_arr.append((filenames_db, legend_names, common_path))

    filenames_db = ['results_amber_trp_300_2_fixed.sqlite3', 'results_charm_trp_300_2_fixed.sqlite3', 'results_gromos_trp_300_2_fixed.sqlite3', 'results_opls_trp_300_2_fixed.sqlite3']
    # legend_names = ['TRP amber_2', 'TRP charm_2', 'TRP gromos_2', 'TRP opls_2']
    legend_names = ['1L2Y, 2nd run with AMBER ff', '1L2Y, 2nd run with CHARM ff', '1L2Y, 2nd run with GROMOS ff', '1L2Y, 2nd run with OPLS ff']
    common_path = '../trp_all_2_compar'
    batch_arr.append((filenames_db, legend_names, common_path))

    filenames_db = ['results_amber_trp_300_fixed.sqlite3', 'results_charm_trp_300_fixed.sqlite3', 'results_gromos_trp_300_fixed.sqlite3', 'results_opls_trp_300_fixed.sqlite3']
    # legend_names = ['TRP amber_1', 'TRP charm_1', 'TRP gromos_1', 'TRP opls_1']
    legend_names = ['1L2Y, 1st run with AMBER ff', '1L2Y, 1st run with CHARM ff', '1L2Y, 1st run with GROMOS ff', '1L2Y, 1st run with OPLS ff']
    common_path = '../trp_all_1_compar'
    batch_arr.append((filenames_db, legend_names, common_path))

    filenames_db = ['results_amber_trp_300_fixed.sqlite3', 'results_amber_trp_300_2_fixed.sqlite3', 'results_charm_trp_300_fixed.sqlite3', 'results_charm_trp_300_2_fixed.sqlite3', 'results_gromos_trp_300_fixed.sqlite3', 'results_gromos_trp_300_2_fixed.sqlite3', 'results_opls_trp_300_fixed.sqlite3', 'results_opls_trp_300_2_fixed.sqlite3']
    legend_names = ['1L2Y, 1st run with AMBER ff', '1L2Y, 2nd run with AMBER ff', '1L2Y, 1st run with CHARM ff', '1L2Y, 2nd run with CHARM ff', '1L2Y, 1st run with GROMOS ff', '1L2Y, 2nd run with GROMOS ff', '1L2Y, 1st run with OPLS ff', '1L2Y, 2nd run with OPLS ff']
    # legend_names = ['TRP amber_1', 'TRP amber_2', 'TRP charm_1', 'TRP charm_2', 'TRP gromos_1', 'TRP gromos_2', 'TRP opls_1', 'TRP opls_2']
    legend_names = ['1L2Y, 1st run with AMBER ff', '1L2Y, 2nd run with AMBER ff', '1L2Y, 1st run with CHARM ff', '1L2Y, 2nd run with CHARM ff', '1L2Y, 1st run with GROMOS ff', '1L2Y, 2nd run with GROMOS ff', '1L2Y, 1st run with OPLS ff', '1L2Y, 2nd run with OPLS ff']
    common_path = '../trp_all_compar'
    batch_arr.append((filenames_db, legend_names, common_path))

    # # ##################  VIL  #######################

    filenames_db = ['results_amber_vil_300.sqlite3', 'results_charm_vil_300.sqlite3', 'results_gromos_vil_300.sqlite3', 'results_opls_vil_300.sqlite3']
    # legend_names = ['VIL amber', 'VIL charm', 'VIL gromos', 'VIL opls']
    legend_names = ['1YRF with AMBER ff', '1YRF with CHARM ff', '1YRF with GROMOS ff', '1YRF with OPLS ff']
    common_path = '../vil_all_compar'
    batch_arr.append((filenames_db, legend_names, common_path))

    # # ##################  GB1  #######################
    # #
    filenames_db = ['results_amber_gb1_300.sqlite3', 'results_charm_gb1_300.sqlite3', 'results_gromos_gb1_300.sqlite3', 'results_opls_gb1_300.sqlite3']
    # legend_names = ['GB1 amber', 'GB1 charm', 'GB1 gromos', 'GB1 opls']
    legend_names = ['1GB1 with AMBER ff', '1GB1 with CHARM ff', '1GB1 with GROMOS ff', '1GB1 with OPLS ff']
    common_path = '../gb1_all_compar'
    batch_arr.append((filenames_db, legend_names, common_path))


    for filenames_db, legend_names, common_path in batch_arr:
        gen_all(filenames_db, legend_names, common_path)



def gen_all(filenames_db: list, legend_names: list, common_path: str):
    """Takes the tasks and processes them either one by one or in parallel.

    Args:
        :param list filenames_db: list of databases
        :param list legend_names: correct names for DBs
        :param str common_path: where to store plots
    """
    fig_num = 0
    try:
        os.mkdir(common_path)
    except:
        pass
    # mdpi = 400
    #
    # font = {'family': 'serif',
    #         'color':  'darkred',
    #         'weight': 'normal',
    #         'size': 12,
    #         }
    parallel = True  # both work, use parallel to generate everything fast, use debug otherwise
    if parallel:
        pool = mp.Pool(len(['rmsd', 'angl', 'andh', 'and', 'xor']))  # we are IO bound in graphs, no need to use exact number of CPUs mp.cpu_count()
        results1 = pool.starmap_async(guide_metr_usage, [(fig_num, filenames_db, legend_names, guide_metr, common_path) for guide_metr in ['rmsd', 'angl', 'andh', 'and', 'xor']])
        results2 = pool.starmap_async(best_traj, [(fig_num, filenames_db, legend_names, guide_metr, common_path) for guide_metr in ['rmsd', 'angl', 'andh', 'and', 'xor']])
        results1.get()
        results2.get()
        pool.close()
    else:  # then debug
        # for guide_metr in ['rmsd', 'angl', 'andh', 'and', 'xor']:
        #     fig_num = guide_metr_usage(fig_num, filenames_db, legend_names, guide_metr, common_path)

        for guide_metr in ['rmsd', 'angl', 'andh', 'and', 'xor']:
            best_traj(fig_num, filenames_db, legend_names, guide_metr, common_path)


def best_traj(fig_num: int, filenames_db: list, legend_names: list, guide_metr: str, common_path: str):
    """This is just a basic comparison among metrics

    Args:
        :param list fig_num: figure number for matplotlib
        :param list filenames_db: databases with data
        :param list legend_names: database names
        :param str guide_metr:
        :param str common_path:

    """
    print('Working with ', filenames_db, ' guide metr: ', guide_metr, ' common path: ', common_path)
    con_arr = [lite.connect(db_name, check_same_thread=False, isolation_level=None) for db_name in filenames_db]
    cur_arr = [con.cursor() for con in con_arr]

    common_path = os.path.join(common_path, guide_metr)
    try:
        os.mkdir(common_path)
    except:
        pass
    plot_all_best_traj(fig_num, cur_arr, filenames_db, legend_names, guide_metr, common_path)
    plot_sep_best_traj(fig_num, cur_arr, filenames_db, legend_names, guide_metr, common_path)


def plot_all_best_traj(fig_num: int, cur_arr: list, filenames_db: list, legend_names: list, guide_metr: str, common_path: str) -> int:
    """

    Args:
        :param int fig_num:
        :param list cur_arr:
        :param list filenames_db:
        :param list legend_names:
        :param str guide_metr:
        :param str common_path:

    Returns:
        :return: figure number
        :rtype: int
    """
    print('Working with ', filenames_db, ' guide metr: ', guide_metr, ' common path: ', common_path)
    qry = "select a.name from main_storage a where a.{0}_goal_dist= ( select min(b.{0}_goal_dist) from main_storage b)".format(guide_metr)
    result_arr = [cur.execute(qry) for cur in cur_arr]
    fetched_one_arr = [res.fetchone() for res in result_arr]
    names = [all_res[0] for all_res in fetched_one_arr]
    spnames = [name.split('_') for name in names]
    all_prev_names_s = [['\'{}\''.format('_'.join(spname[:i])) for i in range(1, len(spname)+1)] for spname in spnames]
    long_lines = [", ".join(all_prev_names) for all_prev_names in all_prev_names_s]
    qrys = ["select a.rmsd_goal_dist, a.angl_goal_dist, a.andh_goal_dist, a.and_goal_dist, a.xor_goal_dist, a.rmsd_tot_dist, a.angl_tot_dist, a.andh_tot_dist, a.and_tot_dist, a.xor_tot_dist, a.name, a.hashed_name from main_storage a where a.name in ( {1} ) order by a.id".format(guide_metr, long_line) for long_line in long_lines]
    result_arr = list()
    for i, cur in enumerate(cur_arr):
        result_arr.append(cur.execute(qrys[i]))
    fetched_all_arr = [res.fetchall() for res in result_arr]

    rmsd_dist_arr = [[dist[0] for dist in goal_dist] for goal_dist in fetched_all_arr]
    angl_dist_arr = [[dist[1] for dist in goal_dist] for goal_dist in fetched_all_arr]
    andh_dist_arr = [[dist[2] for dist in goal_dist] for goal_dist in fetched_all_arr]
    and_dist_arr = [[dist[3] for dist in goal_dist] for goal_dist in fetched_all_arr]
    xor_dist_arr = [[dist[4] for dist in goal_dist] for goal_dist in fetched_all_arr]


    rmsd_tot_dist_arr = [[dist[5] for dist in goal_dist] for goal_dist in fetched_all_arr]
    angl_tot_dist_arr = [[dist[6] for dist in goal_dist] for goal_dist in fetched_all_arr]
    andh_tot_dist_arr = [[dist[7] for dist in goal_dist] for goal_dist in fetched_all_arr]
    and_tot_dist_arr = [[dist[8] for dist in goal_dist] for goal_dist in fetched_all_arr]
    xor_tot_dist_arr = [[dist[9] for dist in goal_dist] for goal_dist in fetched_all_arr]

    goal_dist = [rmsd_dist_arr, angl_dist_arr, andh_dist_arr, and_dist_arr, xor_dist_arr]
    tot_dist = [rmsd_tot_dist_arr, angl_tot_dist_arr, andh_tot_dist_arr, and_tot_dist_arr, xor_tot_dist_arr]
    metrics = ['rmsd', 'angl', 'andh', 'and', 'xor']
    metr_units = {'rmsd': 'Å', 'angl': '', 'andh': 'contacts', 'and': 'contacts', 'xor': 'contacts'}



    for i, dist_arr in enumerate(goal_dist):  # iterate over metric
        max_len = max([len(arr) for arr in dist_arr])
        max_pos_metr_val = max([max(arr) for arr in dist_arr])
        init_metr = dist_arr[0][0]

        ax_prop = {"min_lim_x": 0 - max_len / 80, "max_lim_x": max_len + max_len / 80, "min_lim_y": 0 - max_pos_metr_val / 80, "max_lim_y": max_pos_metr_val + max_pos_metr_val / 80, "min_ax_x": 0,
                   "max_ax_x": max_len + max_len / 80, "min_ax_y": 0, "max_ax_y": max_pos_metr_val + max_pos_metr_val / 80, "ax_step_x": math.floor(max_len / 16), "ax_step_y": max_pos_metr_val / 20}
        if metr_units[metrics[i]] == 'contacts':
            extra_line = [{"ax_type": 'hor', "val": init_metr, "name": "Initial {} metric ({} {})".format(metrics[i].upper(), int(init_metr), metr_units[metrics[i]]), "col": "darkmagenta"}]
        else:
            extra_line = [{"ax_type": 'hor', "val": init_metr, "name": "Initial {} metric ({:3.2f} {})".format(metrics[i].upper(), init_metr, metr_units[metrics[i]]), "col": "darkmagenta"}]
        if metrics[i] == 'rmsd':
            extra_line.append({"ax_type": 'hor', "val": 2.7, "name": "Typical folding mark (2.7 {})".format(metr_units[metrics[i]]), "col": "midnightblue"})
        title = "{} version of the best trajectory | {} view".format(guide_metr, metrics[i])
        filename = "{}_version_of_best_traj_{}".format(guide_metr, metrics[i])
        filename = os.path.join(common_path, filename)
        fig_num = single_plot(fig_num, ax_prop, dist_arr, None, legend_names.copy(), '-', 1, bsf=False, rev=False, extra_line=extra_line, shrink=True, xlab="Steps (20ps each)", ylab="Distance to the goal, {}".format(metr_units[metrics[i]]), title=title, filename=filename)

        max_tot_dist = max([dist[-1] for dist in tot_dist[i]])
        ax_prop = {"min_lim_x": max_pos_metr_val + max_pos_metr_val / 80, "max_lim_x": 0 - max_pos_metr_val / 80, "min_lim_y": 0 - max_tot_dist / 80, "max_lim_y": max_tot_dist + max_tot_dist / 80, "min_ax_x": 0, "max_ax_x": max_pos_metr_val + max_pos_metr_val / 80, "min_ax_y": 0, "max_ax_y": max_tot_dist + max_tot_dist / 80, "ax_step_x": max_pos_metr_val / 20, "ax_step_y": max_tot_dist / 20}
        if metr_units[metrics[i]] == 'contacts':
            extra_line = [{"ax_type": 'ver', "val": init_metr, "name": "Initial {} metric ({} {})".format(metrics[i].upper(), int(init_metr), metr_units[metrics[i]]), "col": "darkmagenta"}]
        else:
            extra_line = [{"ax_type": 'ver', "val": init_metr, "name": "Initial {} metric ({:3.2f} {})".format(metrics[i].upper(), init_metr, metr_units[metrics[i]]), "col": "darkmagenta"}]
        if metrics[i] == 'rmsd':
            extra_line.append({"ax_type": 'ver', "val": 2.7, "name": "Typical folding mark (2.7 {})".format(metr_units[metrics[i]]), "col": "midnightblue"})
        title = "{} version of the best trajectory vs distance traveled | {} view".format(guide_metr, metrics[i])
        filename = '{}_version_of_best_traj_{}_vs_dist'.format(guide_metr, metrics[i])
        filename = os.path.join(common_path, filename)
        fig_num = single_plot(fig_num, ax_prop, dist_arr, tot_dist[i], legend_names.copy(), '-', 1, bsf=False, rev=True, extra_line=extra_line, shrink=False, xlab="Distance to the goal, {}".format(metr_units[metrics[i]]), ylab="Past distance, {}".format(metr_units[metrics[i]]), title=title, filename=filename)

        for j in range(len(dist_arr)):  # iterate over dbs
            max_pos_metr_val = max(dist_arr[j])
            ax_prop = {"min_lim_x": 0 - max_len / 80, "max_lim_x": max_len + max_len / 80, "min_lim_y": 0, "max_lim_y": max_pos_metr_val + max_pos_metr_val / 80, "min_ax_x": 0,
                       "max_ax_x": max_len + max_len / 80, "min_ax_y": 0, "max_ax_y": max_pos_metr_val + max_pos_metr_val / 80, "ax_step_x": max_len / 16, "ax_step_y": max_pos_metr_val / 20}
            if metr_units[metrics[i]] == 'contacts':
                extra_line = [{"ax_type": 'hor', "val": init_metr, "name": "Initial {} metric ({} {})".format(metrics[i].upper(), int(init_metr), metr_units[metrics[i]]), "col": "darkmagenta"},
                              {"ax_type": 'hor', "val": min(dist_arr[j]), "name": "The lowest {} metric ({} {})".format(metrics[i].upper(), int(min(dist_arr[j])), metr_units[metrics[i]]), "col": "darkgreen"}]
            else:
                extra_line = [{"ax_type": 'hor', "val": init_metr, "name": "Initial {} metric ({:3.2f} {})".format(metrics[i].upper(), init_metr, metr_units[metrics[i]]), "col": "darkmagenta"},
                              {"ax_type": 'hor', "val": min(dist_arr[j]), "name": "The lowest {} metric ({:3.2f} {})".format(metrics[i].upper(), min(dist_arr[j]), metr_units[metrics[i]]), "col": "darkgreen"}]

            if metrics[i] == 'rmsd':
                extra_line.append({"ax_type": 'hor', "val": 2.7, "name": "Typical folding mark (2.7 {})".format(metr_units[metrics[i]]), "col": "midnightblue"})
            title = "{} version of the best trajectory | {} view".format(guide_metr, metrics[i])
            filename = "{}_version_of_best_traj_{}_only_{}".format(guide_metr, metrics[i], filenames_db[j].split('.')[0])
            filename = os.path.join(common_path, filename)
            fig_num = single_plot(fig_num, ax_prop, [dist_arr[j]], None, [legend_names[j]], '-', 1, bsf=False, rev=False, extra_line=extra_line, shrink=True, xlab="Steps (20ps each)", ylab="Distance to the goal, {}".format(metr_units[metrics[i]]), title=title, filename=filename)

            max_tot_dist = max([dist[-1] for dist in [tot_dist[i][j]]])
            ax_prop = {"min_lim_x": max_pos_metr_val + max_pos_metr_val / 80, "max_lim_x": 0 - max_pos_metr_val / 80, "min_lim_y": 0 - max_tot_dist / 80, "max_lim_y": max_tot_dist + max_tot_dist / 80, "min_ax_x": 0,
                       "max_ax_x": max_pos_metr_val + max_pos_metr_val / 80, "min_ax_y": 0, "max_ax_y": max_tot_dist + max_tot_dist / 80, "ax_step_x": max_pos_metr_val / 20, "ax_step_y": max_tot_dist / 20}
            if metr_units[metrics[i]] == 'contacts':
                extra_line = [{"ax_type": 'ver', "val": init_metr, "name": "Initial {} metric ({} {})".format(metrics[i].upper(), int(init_metr), metr_units[metrics[i]]), "col": "darkmagenta"},
                              {"ax_type": 'ver', "val": min(dist_arr[j]), "name": "The lowest {} metric ({} {})".format(metrics[i].upper(), int(min(dist_arr[j])), metr_units[metrics[i]]), "col": "darkgreen"}]
            else:
                extra_line = [{"ax_type": 'ver', "val": init_metr, "name": "Initial {} metric ({:3.2f} {})".format(metrics[i].upper(), init_metr, metr_units[metrics[i]]), "col": "darkmagenta"},
                              {"ax_type": 'ver', "val": min(dist_arr[j]), "name": "The lowest {} metric ({:3.2f} {})".format(metrics[i].upper(), min(dist_arr[j]), metr_units[metrics[i]]), "col": "darkgreen"}]
            if metrics[i] == 'rmsd':
                extra_line.append({"ax_type": 'ver', "val": 2.7, "name": "Typical folding mark (2.7 {})".format(metr_units[metrics[i]]), "col": "midnightblue"})
            title = "{} version of the best trajectory vs distance traveled | {} view".format(guide_metr, metrics[i])
            filename = '{}_version_of_best_traj_{}_vs_dist_only_{}'.format(guide_metr, metrics[i], filenames_db[j].split('.')[0])
            filename = os.path.join(common_path, filename)
            fig_num = single_plot(fig_num, ax_prop, [dist_arr[j]], [tot_dist[i][j]], [legend_names[j]], '-', 1, bsf=False, rev=True, extra_line=extra_line, shrink=False, xlab="Distance to the goal, {}".format(metr_units[metrics[i]]), ylab="Past distance, {}".format(metr_units[metrics[i]]), title=title, filename=filename)

            max_pos_metr_val = dist_arr[j][0]
            min_pos_metr_val = dist_arr[j][-1]
            if min_pos_metr_val > max_pos_metr_val:
                min_pos_metr_val, max_pos_metr_val = max_pos_metr_val, min_pos_metr_val


            loc_len = len(dist_arr[j])
            for k in range(len(goal_dist)):
                if i != k:
                    max_pos_metr2_val = goal_dist[k][j][0]
                    min_pos_metr2_val = goal_dist[k][j][-1]
                    if max_pos_metr2_val < min_pos_metr2_val:
                        max_pos_metr2_val, min_pos_metr2_val = min_pos_metr2_val, max_pos_metr2_val

                    divider_min = 15.0
                    divider_max = 10.0

                    while divider_min > 0.1:
                        if (min_pos_metr2_val - (max_pos_metr2_val - min_pos_metr2_val) / divider_min) < min(goal_dist[k][j]) and min_pos_metr_val - (max_pos_metr_val - min_pos_metr_val) / divider_min < min(
                                dist_arr[j]):
                            break
                        divider_min -= 0.05

                    while divider_max > 0.1:
                        if (max_pos_metr2_val + (max_pos_metr2_val - min_pos_metr2_val) / divider_max) > max(goal_dist[k][j]) and max_pos_metr_val + (max_pos_metr_val - min_pos_metr_val) / divider_max > max(
                                dist_arr[j]):
                            break
                        divider_max -= 0.05

                    ax_prop = {"min_lim_x": 0 - loc_len / 80, "max_lim_x": loc_len + loc_len / 80, "min_lim_y": min_pos_metr_val - (max_pos_metr_val - min_pos_metr_val) / divider_min,
                               "max_lim_y": max_pos_metr_val + (max_pos_metr_val - min_pos_metr_val) / divider_max, "min_ax_x": 0,
                               "max_ax_x": loc_len + loc_len / 80, "min_ax_y": min_pos_metr_val - (max_pos_metr_val - min_pos_metr_val) / divider_min, "max_ax_y": max_pos_metr_val + (max_pos_metr_val - min_pos_metr_val) / divider_max,
                               "ax_step_x": math.floor(loc_len / 16), "ax_step_y": (max_pos_metr_val + (max_pos_metr_val - min_pos_metr_val) / divider_max - min_pos_metr_val + (max_pos_metr_val - min_pos_metr_val) / divider_min) / 20}
                    ax2_prop = {"min_lim_y": min_pos_metr2_val - (max_pos_metr2_val - min_pos_metr2_val) / divider_min, "max_lim_y": max_pos_metr2_val + (max_pos_metr2_val - min_pos_metr2_val) / divider_max,
                                "min_ax_y": min_pos_metr2_val - (max_pos_metr2_val - min_pos_metr2_val) / divider_min, "max_ax_y": max_pos_metr2_val + (max_pos_metr2_val - min_pos_metr2_val) / divider_max,  "ax_step_y": (max_pos_metr2_val + (max_pos_metr2_val - min_pos_metr2_val) / divider_max - min_pos_metr2_val + (max_pos_metr2_val - min_pos_metr2_val) / divider_min) / 20,
                                "label": "Distance to the goal ({}), {}".format(metrics[k].upper(), metr_units[metrics[k]]), "line_name": '{} ({})'.format(legend_names[j], metrics[k].upper())}
                    if metr_units[metrics[i]] == 'contacts':
                        extra_line = [
                            {"ax_type": 'hor', "val": init_metr, "name": "Initial {} metric ({} {})".format(metrics[i].upper(), int(init_metr), metr_units[metrics[i]]), "col": "darkmagenta"},
                            {"ax_type": 'hor', "val": min(dist_arr[j]), "name": "The lowest {} metric ({} {})".format(metrics[i].upper(), int(min(dist_arr[j])), metr_units[metrics[i]]), "col": "darkgreen"}]
                    else:
                        extra_line = [
                            {"ax_type": 'hor', "val": init_metr, "name": "Initial {} metric ({:3.2f} {})".format(metrics[i].upper(), init_metr, metr_units[metrics[i]]), "col": "darkmagenta"},
                            {"ax_type": 'hor', "val": min(dist_arr[j]), "name": "The lowest {} metric ({:3.2f} {})".format(metrics[i].upper(), min(dist_arr[j]), metr_units[metrics[i]]), "col": "darkgreen"}]
                    if metrics[i] == 'rmsd':
                        extra_line.append({"ax_type": 'hor', "val": 2.7, "name": "Typical folding mark (2.7 {})".format(metr_units[metrics[i]]), "col": "midnightblue"})
                    title = "{} version of the best trajectory | {} view vs {} view".format(guide_metr, metrics[i], metrics[k])
                    filename = "{}_version_of_best_traj_{}_only_{}_vs_{}".format(guide_metr, metrics[i], filenames_db[j].split('.')[0], metrics[k])
                    filename = os.path.join(common_path, filename)
                    try:
                        fig_num = single_plot(fig_num, ax_prop, [dist_arr[j]], None, ['{} ({})'.format(legend_names[j], metrics[i].upper())], '-', 1, bsf=False, rev=False, extra_line=extra_line, shrink=True, xlab="Steps (20ps each)",
                                          ylab="Distance to the goal ({}), {}".format(metrics[i].upper(), metr_units[metrics[i]]), title=title, filename=filename, second_ax=ax2_prop, sec_arr=goal_dist[k][j])
                    except Exception as e:
                        print('Error in generation of {}'.format(filename))

            loc_len = len(dist_arr[j])
            # prot_name, ff = legend_names[j].split(' ')
            if 'AMBER' in legend_names[j].upper():
                ff = 'amber'
            elif 'CHARM' in legend_names[j].upper():
                ff = 'charm'
            elif 'GROMOS' in legend_names[j].upper():
                ff = 'gromos'
            elif 'OPLS' in legend_names[j].upper():
                ff = 'opls'

            if 'TRP' in legend_names[j].upper() or '1L2Y' in legend_names[j].upper():
                prot_name = 'TRP'
            elif 'VIL' in legend_names[j].upper() or '1YRF' in legend_names[j].upper():
                prot_name = 'VIL'
            elif 'GB1' in legend_names[j].upper():
                prot_name = 'GB1'

            if '2ND' in legend_names[j].upper():
                rn = 2
            elif '1ST' in legend_names[j].upper():
                rn = 1
            else:
                rn = None
            # if '_' in ff:
            #     ff, rn = ff.split('_')
            path_to_ener = "/home/vanya/Documents/Phillips/GMDA/Latest_results"
            path_to_ener1 = os.path.join(path_to_ener, prot_name)
            if rn is not None:
                path_to_ener1 = os.path.join(path_to_ener1, "run_{}".format(rn))
            # path_to_ener2 = os.path.join(path_to_ener1, ff, 'LJ_energy')
            # np_ener_file = os.path.join(path_to_ener2, '{}_combined_energy_best_full_step.npy'.format(guide_metr))
            # ener_arr = np.load(np_ener_file).swapaxes(0, 1)[1]
            # ener_arr = ener_arr[-loc_len:]  # trim, so we have same number of steps
            # if len(ener_arr) != loc_len:
            #     print('kva')
            #
            # max_pos_metr2_val = ener_arr[0]
            # min_pos_metr2_val = ener_arr[-1]
            #
            # ax_prop = {"min_lim_x": 0 - loc_len / 80, "max_lim_x": loc_len + loc_len / 80, "min_lim_y": min_pos_metr_val - (max_pos_metr_val - min_pos_metr_val) / 5.0,
            #            "max_lim_y": max_pos_metr_val + (max_pos_metr_val - min_pos_metr_val) / 10, "min_ax_x": 0,
            #            "max_ax_x": loc_len + loc_len / 80, "min_ax_y": min_pos_metr_val - (max_pos_metr_val - min_pos_metr_val) / 5.0,
            #            "max_ax_y": max_pos_metr_val + (max_pos_metr_val - min_pos_metr_val) / 10,
            #            "ax_step_x": loc_len / 16, "ax_step_y": (max_pos_metr_val - min_pos_metr_val) / 20}
            # ax2_prop = {"min_lim_y": min_pos_metr2_val - (max_pos_metr2_val - min_pos_metr2_val) / 5.0, "max_lim_y": max_pos_metr2_val + (max_pos_metr2_val - min_pos_metr2_val) / 10,
            #             "min_ax_y": min_pos_metr2_val - (max_pos_metr2_val - min_pos_metr2_val) / 5.0, "max_ax_y": max_pos_metr2_val + (max_pos_metr2_val - min_pos_metr2_val) / 10,
            #             "ax_step_y": (max_pos_metr2_val - min_pos_metr2_val) / 20,
            #             "label": "LJ energy, {}".format('kJ/mol'), "line_name": 'LJ:SR interaction energy ({})'.format('kJ/mol')}
            # extra_line = [{"ax_type": 'hor', "val": init_metr, "name": "initial {} metric ({:3.2f} {})".format(metrics[i], init_metr, metr_units[metrics[i]]), "col": "darkmagenta"}]
            # if metrics[i] == 'rmsd':
            #     extra_line.append({"ax_type": 'hor', "val": 2.7, "name": "typical folding mark (2.7 {})".format(metr_units[metrics[i]]), "col": "midnightblue"})
            # title = "{} version of the best trajectory | {} view vs LJ:SR view".format(guide_metr, metrics[i])
            # filename = "{}_version_of_best_traj_{}_only_{}_vs_{}".format(guide_metr, metrics[i], filenames_db[j].split('.')[0], 'lj_energy')
            # filename = os.path.join(common_path, filename)
            # fig_num = single_plot(fig_num, ax_prop, [dist_arr[j]], None, ['{} ({})'.format(legend_names[j], metrics[i])], '-', 1, bsf=False, rev=False, extra_line=extra_line, shrink=True,
            #                       xlab="steps (20ps each)",
            #                       ylab="to goal ({}), {}".format(metrics[i], metr_units[metrics[i]]), title=title, filename=filename, second_ax=ax2_prop, sec_arr=ener_arr)
            #
            #
            # path_to_ener2 = os.path.join(path_to_ener1, ff, 'CL_energy')
            # np_ener_file = os.path.join(path_to_ener2, '{}_combined_energy_best_full_step.npy'.format(guide_metr))
            # ener_arr = np.load(np_ener_file).swapaxes(0, 1)[1]
            # ener_arr = ener_arr[-loc_len:]  # trim, so we have same number of steps
            #
            # max_pos_metr2_val = ener_arr[0]
            # min_pos_metr2_val = ener_arr[-1]
            #
            # ax_prop = {"min_lim_x": 0 - loc_len / 80, "max_lim_x": loc_len + loc_len / 80, "min_lim_y": min_pos_metr_val - (max_pos_metr_val - min_pos_metr_val) / 5.0,
            #            "max_lim_y": max_pos_metr_val + (max_pos_metr_val - min_pos_metr_val) / 10, "min_ax_x": 0,
            #            "max_ax_x": loc_len + loc_len / 80, "min_ax_y": min_pos_metr_val - (max_pos_metr_val - min_pos_metr_val) / 5.0,
            #            "max_ax_y": max_pos_metr_val + (max_pos_metr_val - min_pos_metr_val) / 10,
            #            "ax_step_x": loc_len / 16, "ax_step_y": (max_pos_metr_val - min_pos_metr_val) / 20}
            # ax2_prop = {"min_lim_y": min_pos_metr2_val - (max_pos_metr2_val - min_pos_metr2_val) / 5.0, "max_lim_y": max_pos_metr2_val + (max_pos_metr2_val - min_pos_metr2_val) / 10,
            #             "min_ax_y": min_pos_metr2_val - (max_pos_metr2_val - min_pos_metr2_val) / 5.0, "max_ax_y": max_pos_metr2_val + (max_pos_metr2_val - min_pos_metr2_val) / 10,
            #             "ax_step_y": (max_pos_metr2_val - min_pos_metr2_val) / 20,
            #             "label": "CL energy, {}".format('kJ/mol'), "line_name": 'CL:SR interaction energy ({})'.format('kJ/mol')}
            # extra_line = [{"ax_type": 'hor', "val": init_metr, "name": "initial {} metric ({:3.2f} {})".format(metrics[i], init_metr, metr_units[metrics[i]]), "col": "darkmagenta"}]
            # if metrics[i] == 'rmsd':
            #     extra_line.append({"ax_type": 'hor', "val": 2.7, "name": "typical folding mark (2.7 {})".format(metr_units[metrics[i]]), "col": "midnightblue"})
            # title = "{} version of the best trajectory | {} view vs CL:SR view".format(guide_metr, metrics[i])
            # filename = "{}_version_of_best_traj_{}_only_{}_vs_{}".format(guide_metr, metrics[i], filenames_db[j].split('.')[0], 'cl_energy')
            # filename = os.path.join(common_path, filename)
            # fig_num = single_plot(fig_num, ax_prop, [dist_arr[j]], None, ['{} ({})'.format(legend_names[j], metrics[i])], '-', 1, bsf=False, rev=False, extra_line=extra_line, shrink=True,
            #                       xlab="steps (20ps each)",
            #                       ylab="to goal ({}), {}".format(metrics[i], metr_units[metrics[i]]), title=title, filename=filename, second_ax=ax2_prop, sec_arr=ener_arr)




            path_to_ener2 = os.path.join(path_to_ener1, ff, 'PT_energy')
            np_ener_file = os.path.join(path_to_ener2, '{}_correct_index_energy.npy'.format(guide_metr))
            ener_arr = np.load(np_ener_file).swapaxes(0, 1)[1]
            ener_arr = ener_arr[-loc_len:]  # trim, so we have same number of steps

            max_pos_metr2_val = ener_arr[0]
            min_pos_metr2_val = ener_arr[-1]

            divider_min = 5.0
            divider_max = 10.0

            while divider_min > 0.1:
                if (min_pos_metr2_val - (max_pos_metr2_val - min_pos_metr2_val) / divider_min) < min(ener_arr) and min_pos_metr_val - (max_pos_metr_val - min_pos_metr_val) / divider_min < min(
                        dist_arr[j]):
                    break
                divider_min -= 0.05

            while divider_max > 0.1:
                if (max_pos_metr2_val + (max_pos_metr2_val - min_pos_metr2_val) / divider_max) > max(ener_arr) and max_pos_metr_val + (max_pos_metr_val - min_pos_metr_val) / divider_max > max(
                        dist_arr[j]):
                    break
                divider_max -= 0.05

            ax_prop = {"min_lim_x": 0 - loc_len / 80, "max_lim_x": loc_len + loc_len / 80, "min_lim_y": min_pos_metr_val - (max_pos_metr_val - min_pos_metr_val) / divider_min,
                       "max_lim_y": max_pos_metr_val + (max_pos_metr_val - min_pos_metr_val) / divider_max, "min_ax_x": 0,
                       "max_ax_x": loc_len + loc_len / 80, "min_ax_y": min_pos_metr_val - (max_pos_metr_val - min_pos_metr_val) / divider_min,
                       "max_ax_y": max_pos_metr_val + (max_pos_metr_val - min_pos_metr_val) / divider_max,
                       "ax_step_x": math.floor(loc_len / 16), "ax_step_y": (max_pos_metr_val + (max_pos_metr_val - min_pos_metr_val) / divider_max - min_pos_metr_val + (max_pos_metr_val - min_pos_metr_val) / divider_min) / 20}
            ax2_prop = {"min_lim_y": min_pos_metr2_val - (max_pos_metr2_val - min_pos_metr2_val) / divider_min, "max_lim_y": max_pos_metr2_val + (max_pos_metr2_val - min_pos_metr2_val) / divider_max,
                        "min_ax_y": min_pos_metr2_val - (max_pos_metr2_val - min_pos_metr2_val) / divider_min, "max_ax_y": max_pos_metr2_val + (max_pos_metr2_val - min_pos_metr2_val) / divider_max,
                        "ax_step_y": (max_pos_metr2_val + (max_pos_metr2_val - min_pos_metr2_val) / divider_max - min_pos_metr2_val + (max_pos_metr2_val - min_pos_metr2_val) / divider_min) / 20,
                        "label": "Potential energy, {}".format('kJ/mol'), "line_name": 'Potential energy ({})'.format('kJ/mol')}
            if metr_units[metrics[i]] == 'contacts':
                extra_line = [
                    {"ax_type": 'hor', "val": init_metr, "name": "Initial {} metric ({} {})".format(metrics[i].upper(), int(init_metr), metr_units[metrics[i]]), "col": "darkmagenta"},
                    {"ax_type": 'hor', "val": min(dist_arr[j]), "name": "The lowest {} metric ({} {})".format(metrics[i].upper(), int(min(dist_arr[j])), metr_units[metrics[i]]), "col": "darkgreen"}]
            else:
                extra_line = [
                    {"ax_type": 'hor', "val": init_metr, "name": "Initial {} metric ({:3.2f} {})".format(metrics[i].upper(), init_metr, metr_units[metrics[i]]), "col": "darkmagenta"},
                    {"ax_type": 'hor', "val": min(dist_arr[j]), "name": "The lowest {} metric ({:3.2f} {})".format(metrics[i].upper(), min(dist_arr[j]), metr_units[metrics[i]]), "col": "darkgreen"}]
            if metrics[i] == 'rmsd':
                extra_line.append({"ax_type": 'hor', "val": 2.7, "name": "Typical folding mark (2.7 {})".format(metr_units[metrics[i]]), "col": "midnightblue"})
            title = "{} version of the best trajectory | {} view vs Potential energy view".format(guide_metr, metrics[i])
            filename = "{}_version_of_best_traj_{}_only_{}_vs_{}".format(guide_metr, metrics[i], filenames_db[j].split('.')[0], 'pt_energy')
            filename = os.path.join(common_path, filename)
            fig_num = single_plot(fig_num, ax_prop, [dist_arr[j]], None, ['{} ({})'.format(legend_names[j], metrics[i].upper())], '-', 1, bsf=False, rev=False, extra_line=extra_line, shrink=True,
                                  xlab="Steps (20ps each)",
                                  ylab="Distance to the goal ({}), {}".format(metrics[i].upper(), metr_units[metrics[i]]), title=title, filename=filename, second_ax=ax2_prop, sec_arr=ener_arr)



    # max_len = max([len(arr) for arr in rmsd_dist_arr])
    # init_metr = rmsd_dist_arr[0][0]
    # metr_units = 'A'
    # ax_prop = {"min_lim_x": 0 - +max_len/80, "max_lim_x": max_len + max_len/80, "min_lim_y": 0 - init_metr/80, "max_lim_y": init_metr + init_metr/80, "min_ax_x": 0, "max_ax_x": max_len + max_len/80, "min_ax_y": 0, "max_ax_y": init_metr+init_metr/80, "ax_step_x": max_len / 16, "ax_step_y": init_metr / 20}
    # extra_line = {"ax_type": 'hor', "val": init_metr, "name": "initial {} metric ({:3.2f} {})".format('rmsd', init_metr, metr_units)}
    # # title = "{} | to goal vs traveled | {} | {} | {}".format(metric, full_cut, same, 'shrink' if shrink else 'unshrink')
    # # filename = "{}_to_goal_vs_traveled_{}_{}_{}".format(metric, full_cut, same, 'shrink' if shrink else 'unshrink')
    # # filename = os.path.join(custom_path, filename)
    # title = 'kva'
    # filename = 'test_best'
    # fig_num = single_plot(fig_num, ax_prop, rmsd_dist_arr, None, legend_names.copy(), '-', 1, bsf=False, rev=False, extra_line=extra_line, shrink=True, xlab="steps (20ps each)", ylab="to goal, {}".format(metr_units), title=title, filename=filename)
    #
    # max_tot_dist = max([dist[-1] for dist in rmsd_tot_dist_arr])
    # # ax_prop = {"min_lim_x": 0 - +max_len/80, "max_lim_x": max_tot_dist + max_tot_dist/80, "min_lim_y": 0 - init_metr/80, "max_lim_y": init_metr + init_metr/80, "min_ax_x": 0, "max_ax_x": max_tot_dist + max_tot_dist/80, "min_ax_y": 0, "max_ax_y": init_metr+init_metr/80, "ax_step_x": max_tot_dist / 16, "ax_step_y": init_metr / 20}
    # ax_prop = {"min_lim_x": init_metr + init_metr / 80, "max_lim_x": 0 - init_metr / 80, "min_lim_y": 0 - +max_len / 80, "max_lim_y": max_tot_dist + max_tot_dist / 80, "min_ax_x": 0,
    #            "max_ax_x": init_metr + init_metr / 80, "min_ax_y": 0, "max_ax_y": max_tot_dist + max_tot_dist / 80, "ax_step_x": init_metr / 20, "ax_step_y": max_tot_dist / 16}
    # extra_line = {"ax_type": 'ver', "val": init_metr, "name": "initial {} metric ({:3.2f} {})".format('rmsd', init_metr, metr_units)}
    # # title = "{} | to goal vs traveled | {} | {} | {}".format(metric, full_cut, same, 'shrink' if shrink else 'unshrink')
    # # filename = "{}_to_goal_vs_traveled_{}_{}_{}".format(metric, full_cut, same, 'shrink' if shrink else 'unshrink')
    # # filename = os.path.join(custom_path, filename)
    # title = 'kva'
    # filename = 'test_best'
    # fig_num = single_plot(fig_num, ax_prop, rmsd_dist_arr, rmsd_tot_dist_arr, legend_names.copy(), '-', 1, bsf=False, rev=True, extra_line=extra_line, shrink=False, xlab="to goal, {}".format(metr_units), ylab="steps (20ps each)", title=title, filename=filename)





def plot_sep_best_traj(fig_num, cur_arr, filenames_db, legend_names, guide_metr, common_path):
    pass


def guide_metr_usage(fig_num: int, filenames_db: list, legend_names: list, guide_metr: str, common_path: str) -> int:
    """

    Args:
        :param int fig_num: figure number, it should not matter, since we close all figures regularly
        :param list filenames_db: database names
        :param list legend_names: proper database description
        :param str guide_metr: main metric for the plot
        :param str common_path: where to store plots

    Returns:
        :return: figure number, it should not matter, since we close all figures regularly
    """

    con_arr = [lite.connect(db_name, check_same_thread=False, isolation_level=None) for db_name in filenames_db]
    cur_arr = [con.cursor() for con in con_arr]

    common_path = os.path.join(common_path, guide_metr)
    try:
        os.mkdir(common_path)
    except:
        pass

    fig_num, init_rmsd = plot_all_metrics(fig_num, cur_arr, filenames_db, legend_names, guide_metr, common_path)

    for partial_metr in ["RMSD", "ANGL", "AND_H", "AND", "XOR"]:
        pers_path = os.path.join(common_path, partial_metr)
        try:
            os.mkdir(pers_path)
        except:
            pass
        fig_num = plot_only_one_metric(fig_num, cur_arr, filenames_db, init_rmsd, legend_names, partial_metr, guide_metr, pers_path)

    [con.close() for con in con_arr]
    return fig_num


def plot_all_metrics(fig_num: int, cur_arr: list, filenames_db: list, legend_names: list, guide_metr: str, common_path: str) -> int:
    """General force field comparison: sampling, best_so_far, dist traveled

    Args:
        :param int fig_num: figure number, it should not matter, since we close all figures regularly
        :param list cur_arr:
        :param list filenames_db:
        :param list legend_names:
        :param str guide_metr:
        :param str common_path:

    Returns:
        :return: figure number, it should not matter, since we close all figures regularly
    """
    best_metr_dic = {'rmsd': 'bsfr', 'angl': 'bsfn', 'andh': 'bsfh', 'and': 'bsfa', 'xor': 'bsfx'}
    metr_units = {'rmsd': 'Å', 'angl': '', 'andh': 'contacts', 'and': 'contacts', 'xor': 'contacts'}
    qry = 'SELECT a.{}_goal_dist FROM main_storage a join visited b on a.id=b.id order by b.vid'.format(guide_metr)
    result_arr = [cur.execute(qry) for cur in cur_arr]
    fetched_all_arr = [res.fetchall() for res in result_arr]
    filt_res_arr = [[dist[0] for dist in goal_dist] for goal_dist in fetched_all_arr]
    init_rmsd = filt_res_arr[0][0]
    max_non_init_rmsd = max(max(elem) for elem in filt_res_arr)
    common_point = max([min(elem) for elem in filt_res_arr])

    ind_arr = list()
    for rmsd_for_db in filt_res_arr:
        i = 0
        while common_point < rmsd_for_db[i]:
            i += 1
        ind_arr.append(i)

    # print('To reach common min point of {}A ({})'.format(common_point, guide_metr))
    # for i, db in enumerate(filenames_db):
    #     print('{} : {} steps'.format(db.split('.')[0], ind_arr[i]))



    #     ################# CUT ######################

    # qry = "select a.bsfr, b.rmsd_tot_dist, b.rmsd_goal_dist from log a join main_storage b on a.id=b.id where a.dst='VIZ' and a.bsfr>'{}' order by a.lid".format(common_point)
    qry = "select a.{0}, b.{1}_tot_dist, b.{1}_goal_dist, c.vid from main_storage b join visited c on c.id=b.id join (select id, {0} from log where dst='VIZ' group by id) a on a.id=b.id where a.{0}>'{2}'  order by c.vid".format(best_metr_dic[guide_metr], guide_metr, common_point)
    result_arr = [cur.execute(qry) for cur in cur_arr]
    [res.fetchone() for res in result_arr]
    fetched_all_arr = [res.fetchall() for res in result_arr]
    bsf_arr = [[dist[0] for dist in goal_dist] for goal_dist in fetched_all_arr]
    for i in range(len(bsf_arr)):
        bsf_arr[i].insert(0, init_rmsd)
    for j in range(len(bsf_arr)):
        for i in range(len(bsf_arr[j]) - 1):
            if bsf_arr[j][i] < bsf_arr[j][i + 1]:
                bsf_arr[j][i+1] = bsf_arr[j][i]
    trav_arr = [[dist[1] for dist in goal_dist] for goal_dist in fetched_all_arr]
    to_goal_arr = [[dist[2] for dist in goal_dist] for goal_dist in fetched_all_arr]

    max_len = max([len(goal_dist) for goal_dist in fetched_all_arr])
    custom_path = '{}/ALL/'.format(common_path)
    try:
        os.mkdir(custom_path)
    except:
        pass

    try:
        max_trav = max([max(elem) for elem in trav_arr])
        custom_path = '{}/ALL/cut/'.format(common_path)
        try:
            os.mkdir(custom_path)
        except:
            pass
        # shrink is True since everything is in order, there is no difference whether to pass index or generate it
        fig_num = plot_set(fig_num, to_goal_arr, legend_names, max_len, max_non_init_rmsd, init_rmsd, bsf_arr, common_point, max_trav, trav_arr, "cut", guide_metr, metr_units[guide_metr], 'all', custom_path, shrink=True)
    except:
        print('Not all trajecotories have a common point', [len(elem) for elem in trav_arr])

    #     ################# FULL ######################

    # qry = "select a.bsfr, b.rmsd_tot_dist, b.rmsd_goal_dist from log a join main_storage b on a.id=b.id where a.dst='VIZ' order by a.lid"
    qry = "select a.{0}, b.{1}_tot_dist, b.{1}_goal_dist, c.vid from main_storage b join visited c on c.id=b.id join (select id, max({0}) as {0} from log where dst='VIZ' group by id) a on a.id=b.id order by c.vid".format(best_metr_dic[guide_metr], guide_metr)
    result_arr = [cur.execute(qry) for cur in cur_arr]
    [res.fetchone() for res in result_arr]
    fetched_all_arr = [res.fetchall() for res in result_arr]
    bsf_arr = [[dist[0] for dist in goal_dist] for goal_dist in fetched_all_arr]
    for i in range(len(bsf_arr)):
        bsf_arr[i].insert(0, init_rmsd)
    for j in range(len(bsf_arr)):
        for i in range(len(bsf_arr[j]) - 1):
            if bsf_arr[j][i] < bsf_arr[j][i + 1]:
                bsf_arr[j][i+1] = bsf_arr[j][i]

    trav_arr = [[dist[1] for dist in goal_dist] for goal_dist in fetched_all_arr]
    to_goal_arr = [[dist[2] for dist in goal_dist] for goal_dist in fetched_all_arr]

    max_len = max([len(goal_dist) for goal_dist in fetched_all_arr])
    max_trav = max([max(elem) for elem in trav_arr])
    common_point = min([min(elem) for elem in filt_res_arr])

    custom_path = '{}/ALL/full/'.format(common_path)
    try:
        os.mkdir(custom_path)
    except:
        pass
    # shrink is True since everything is in order, there is no difference whether to pass index or generate it
    fig_num = plot_set(fig_num, to_goal_arr, legend_names, max_len, max_non_init_rmsd, init_rmsd, bsf_arr, common_point, max_trav, trav_arr, "full", guide_metr, metr_units[guide_metr], 'all', custom_path, shrink=True)


    return fig_num, init_rmsd


def plot_only_one_metric(fig_num: int, cur_arr: list, filenames_db: list, init_rmsd: float, legend_names: list, metric_name: str, guide_metr: str, common_path: str) -> int:
    """

    Args:
        :param int fig_num:
        :param list cur_arr:
        :param list filenames_db:
        :param float init_rmsd:
        :param list legend_names:
        :param str metric_name:
        :param str guide_metr:
        :param str common_path:

    Returns:
        :return: figure number
    """
    best_metr_dic = {'rmsd': 'bsfr', 'angl': 'bsfn', 'andh': 'bsfh', 'and': 'bsfa', 'xor': 'bsfx'}
    metr_units = {'rmsd': 'Å', 'angl': '', 'andh': 'contacts', 'and': 'contacts', 'xor': 'contacts'}
    # qry = "SELECT a.rmsd_goal_dist, b.vid FROM main_storage a join visited b on a.id=b.id join log c on a.id=c.id where c.cur_metr='{}' order by b.vid".format(metric_name)
    qry = "select a.{0}_goal_dist, b.vid from main_storage a join visited b on a.id=b.id join (select id, cur_metr from log where dst='VIZ' group by id) c on c.id=b.id where c.cur_metr='{1}' order by b.vid".format(guide_metr, metric_name)
    result_arr = [cur.execute(qry) for cur in cur_arr]
    fetched_all_arr = [res.fetchall() for res in result_arr]
    filt_res_arr = [[dist[0] for dist in goal_dist] for goal_dist in fetched_all_arr]
    # init_rmsd = filt_res_arr[0][0]
    max_non_init_rmsd = max(max(elem) for elem in filt_res_arr)
    common_point = max([min(elem) for elem in filt_res_arr])

    ind_arr = list()
    for rmsd_for_db in filt_res_arr:
        i = 0
        while common_point < rmsd_for_db[i]:
            i += 1
        ind_arr.append(i)

    # print('To reach common min point of {}A (rmsd)'.format(common_point))
    # for i, db in enumerate(filenames_db):
    #     print('{} : {} steps'.format(db.split('.')[0], ind_arr[i]))


    #     ################# FULL ######################

    # qry = "select a.bsfr, b.rmsd_tot_dist, b.rmsd_goal_dist, c.vid from log a join main_storage b on a.id=b.id join visited c on c.id=a.id where a.dst='VIZ' and a.cur_metr='{}' order by a.lid".format(metric_name)
    qry = "select c.{0}, a.{1}_tot_dist, a.{1}_goal_dist, b.vid from main_storage a join visited b on a.id=b.id join (select id, max({0}) as {0}, cur_metr from log where dst='VIZ' group by id) c on c.id=b.id where c.cur_metr='{2}' order by b.vid".format(best_metr_dic[guide_metr], guide_metr, metric_name)
    result_arr = [cur.execute(qry) for cur in cur_arr]
    [res.fetchone() for res in result_arr]
    fetched_all_arr = [res.fetchall() for res in result_arr]
    bsf_arr = [[dist[0] for dist in goal_dist] for goal_dist in fetched_all_arr]
    for i in range(len(bsf_arr)):
        bsf_arr[i].insert(0, init_rmsd)
    for j in range(len(bsf_arr)):
        for i in range(len(bsf_arr[j]) - 1):
            if bsf_arr[j][i] < bsf_arr[j][i + 1]:
                bsf_arr[j][i+1] = bsf_arr[j][i]
    trav_arr = [[dist[1] for dist in goal_dist] for goal_dist in fetched_all_arr]
    to_goal_arr = [[dist[2] for dist in goal_dist] for goal_dist in fetched_all_arr]
    non_shr = [[dist[3] for dist in goal_dist] for goal_dist in fetched_all_arr]
    # for i in range(len(non_shr)):
    #     non_shr[i].insert(0, 0)

    max_len = max([len(goal_dist) for goal_dist in fetched_all_arr])
    max_trav = max([max(elem) for elem in trav_arr])
    common_point = min([min(elem) for elem in filt_res_arr])
    custom_path = '{}/full/'.format(common_path)
    try:
        os.mkdir(custom_path)
    except:
        pass

    fig_num = plot_set(fig_num, to_goal_arr, legend_names, max_len, max_non_init_rmsd, init_rmsd, bsf_arr, common_point, max_trav, trav_arr, "full", guide_metr, metr_units[guide_metr], metric_name, custom_path, shrink=True)
    max_len = max([max(arr) for arr in non_shr])
    fig_num = plot_set(fig_num, to_goal_arr, legend_names, max_len, max_non_init_rmsd, init_rmsd, bsf_arr, common_point, max_trav, trav_arr, "full", guide_metr, metr_units[guide_metr], metric_name, custom_path, shrink=False, non_shrink_arr=non_shr)

    return fig_num


def plot_set(fig_num: int, to_goal_arr: list, legend_names: list, max_len: float, max_non_init_rmsd: float,
             init_metr: float, bsf_arr: list, common_point: float, max_trav: float, trav_arr: list, full_cut: str,
             metric: str, metr_units: str, same: str, custom_path: str, shrink: bool, non_shrink_arr: list = None) -> int:
    """

    Args:
        :param int fig_num:
        :param list to_goal_arr:
        :param list legend_names:
        :param float max_len:
        :param float max_non_init_rmsd:
        :param float init_metr:
        :param float list bsf_arr:
        :param float common_point:
        :param float max_trav:
        :param list trav_arr:
        :param str full_cut:
        :param str metric:
        :param str metr_units:
        :param str same:
        :param str custom_path:
        :param bool shrink:
        :param list non_shrink_arr:

    Returns:
        :return: fig number
        :rtype: int
    """
    # # #### SHRINK
    # ax_prop = {"min_lim_x": -max_len/80, "max_lim_x": max_len+max_len/80, "min_lim_y": 0, "max_lim_y": max_non_init_rmsd+max_non_init_rmsd/80, "min_ax_x": 0, "max_ax_x": max_len+max_len/80, "min_ax_y": 0, "max_ax_y": max_non_init_rmsd+max_non_init_rmsd/80, "ax_step_x": max_len/16, "ax_step_y": max_non_init_rmsd/20}
    # extra_line = {"ax_type": 'hor', "val": init_rmsd, "name": "init {} ({:3.2f} {})".format(metric, init_rmsd, metr_units)}
    # fig_num = single_plot(fig_num, ax_prop, to_goal_arr, None,            legend_names, '.', 0.3, bsf=False, rev=False, extra_line=extra_line, xlab="steps (20ps each)", ylab="to goal, A", title="{} | to goal vs traveled | {} | {}".format(metric, full_cut, same), filename="{}_to_goal_vs_traveled_{}_{}".format(metric, full_cut, same))  # to goal vs traveled | cut
    #
    # ax_prop = {"min_lim_x": -max_len/80, "max_lim_x": max_len+max_len/80, "min_lim_y": 0, "max_lim_y": max_non_init_rmsd+max_non_init_rmsd/80, "min_ax_x": 0, "max_ax_x": max_len+max_len/80, "min_ax_y": 0, "max_ax_y": max_non_init_rmsd+max_non_init_rmsd/80, "ax_step_x": max_len/16, "ax_step_y": max_non_init_rmsd/20}
    # extra_line = {"ax_type": 'hor', "val": init_rmsd, "name": "init {} ({:3.2f} {})".format(metric, init_rmsd, metr_units)}
    # fig_num = single_plot(fig_num, ax_prop,  bsf_arr,    None,            legend_names, '-', 1,   bsf=True,  rev=False, extra_line=extra_line, xlab="steps (20ps each)", ylab="steps", title="{} | to goal vs best_so_far | {} | {}".format(metric, full_cut, same), filename="{}_to_goal_vs_best_so_far_{}_{}".format(metric, full_cut, same))  # to goal vs best_so_far | cut
    #
    # ax_prop = {"min_lim_x": max_non_init_rmsd, "max_lim_x": common_point-common_point/10, "min_lim_y": -max_len/80, "max_lim_y": max_len+max_len/80, "min_ax_x": common_point, "max_ax_x": max_non_init_rmsd, "min_ax_y": 0, "max_ax_y": max_len+max_len/80, "ax_step_x": (max_non_init_rmsd-common_point)/16, "ax_step_y": max_len/20}
    # extra_line = {"ax_type": 'ver', "val": init_rmsd, "name": "init {} ({:3.2f} {})".format(metric, init_rmsd, metr_units)}
    # fig_num = single_plot(fig_num, ax_prop,   bsf_arr,   None,            legend_names, '-',  1,   bsf=True,  rev=True,  extra_line=extra_line, xlab="to goal, A", ylab="steps", title="{} | best_so_far vs steps | {} | {}".format(metric, full_cut, same), filename="{}_best_so_far_vs_steps_{}_{}".format(metric, full_cut, same))  # best_so_far vs steps | cut

    # #### NO SHRINK
    custom_path = custom_path+'shrink' if shrink else custom_path+'unshrink'
    try:
        os.mkdir(custom_path)
    except:
        pass
    ax_prop = {"min_lim_x": -max_len/80, "max_lim_x": max_len+max_len/80, "min_lim_y": 0, "max_lim_y": max_non_init_rmsd+max_non_init_rmsd/80,
               "min_ax_x": 0, "max_ax_x": max_len+max_len/80, "min_ax_y": 0, "max_ax_y": max_non_init_rmsd+max_non_init_rmsd/80, "ax_step_x": math.floor(max_len/16), "ax_step_y": max_non_init_rmsd/20}
    if metr_units == 'contacts':
        extra_line = [
            {"ax_type": 'hor', "val": init_metr, "name": "Initial {} metric ({} {})".format(metric.upper(), int(init_metr), metr_units), "col": "darkmagenta"},
            {"ax_type": 'hor', "val": min(min(elem) for elem in to_goal_arr), "name": "The lowest {} metric ({} {})".format(metric.upper(), int(min(min(elem) for elem in to_goal_arr)), metr_units), "col": "darkgreen"}]
    else:
        extra_line = [
            {"ax_type": 'hor', "val": init_metr, "name": "Initial {} metric ({:3.2f} {})".format(metric.upper(), init_metr, metr_units), "col": "darkmagenta"},
            {"ax_type": 'hor', "val": min(min(elem) for elem in to_goal_arr), "name": "The lowest {} metric ({:3.2f} {})".format(metric.upper(), min(min(elem) for elem in to_goal_arr), metr_units), "col": "darkgreen"}]
    if metric == 'rmsd':
        extra_line.append({"ax_type": 'hor', "val": 2.7, "name": "Typical folding mark (2.7 {})".format(metr_units), "col": "midnightblue"})
    title = "{} | to goal vs traveled | {} | {} | {}".format(metric, full_cut, same, 'shrink' if shrink else 'unshrink')
    filename = "{}_to_goal_vs_traveled_{}_{}_{}".format(metric, full_cut, same, 'shrink' if shrink else 'unshrink')
    filename = os.path.join(custom_path, filename)
    fig_num = single_plot(fig_num, ax_prop, to_goal_arr, non_shrink_arr,  legend_names.copy(), '.', 0.3, bsf=False, rev=False, extra_line=extra_line, shrink=shrink, xlab="Steps (20ps each)", ylab="Distance to the goal, {}".format(metr_units), title=title, filename=filename)  # to goal vs traveled | cut

    for i in range(len(to_goal_arr)):
        ff = legend_names[i].split('with')[1].split('ff')[0].strip()
        title = "{} | to goal vs traveled | {} | {} | {} | {}".format(metric, full_cut, same, 'shrink' if shrink else 'unshrink', ff)
        filename = "{}_to_goal_vs_traveled_{}_{}_{}_{}".format(metric, full_cut, same, 'shrink' if shrink else 'unshrink', ff)
        filename = os.path.join(custom_path, filename)
        extra_line[1]["val"] = min(to_goal_arr[i])
        if metr_units == 'contacts':
            extra_line[1]["name"] = "The lowest {} metric ({} {})".format(metric.upper(), int(min(to_goal_arr[i])), metr_units)
        else:
            extra_line[1]["name"] = "The lowest {} metric ({:3.2f} {})".format(metric.upper(), min(to_goal_arr[i]), metr_units)
        fig_num = single_plot(fig_num, ax_prop, [to_goal_arr[i],], [non_shrink_arr[i],] if non_shrink_arr is not None else None, [legend_names[i],].copy(), '.', 0.3, bsf=False, rev=False, extra_line=extra_line, shrink=shrink, xlab="Steps (20ps each)",
                              ylab="Distance to the goal, {}".format(metr_units), title=title, filename=filename)  # to goal vs traveled | cut

    if shrink:
        ax_prop = {"min_lim_x": max_non_init_rmsd, "max_lim_x": common_point-common_point/20, "min_lim_y": -max_trav/80, "max_lim_y": max_trav+max_trav/80,
                   "min_ax_x": common_point, "max_ax_x": max_non_init_rmsd, "min_ax_y": 0, "max_ax_y": max_trav+max_trav/80, "ax_step_x": (max_non_init_rmsd-common_point)/20, "ax_step_y": max_trav/20}
        if metr_units == 'contacts':
            extra_line = [
                {"ax_type": 'ver', "val": init_metr, "name": "Initial {} metric ({} {})".format(metric.upper(), int(init_metr), metr_units), "col": "darkmagenta"},
                {"ax_type": 'ver', "val": min(min(elem) for elem in to_goal_arr), "name": "The lowest {} metric ({} {})".format(metric.upper(), int(min(min(elem) for elem in to_goal_arr)), metr_units), "col": "darkgreen"}]
        else:
            extra_line = [
                {"ax_type": 'ver', "val": init_metr, "name": "Initial {} metric ({:3.2f} {})".format(metric.upper(), init_metr, metr_units), "col": "darkmagenta"},
                {"ax_type": 'ver', "val": min(min(elem) for elem in to_goal_arr), "name": "The lowest {} metric ({:3.2f} {})".format(metric.upper(), min(min(elem) for elem in to_goal_arr), metr_units), "col": "darkgreen"}]
        if metric == 'rmsd':
            extra_line.append({"ax_type": 'hor', "val": 2.7, "name": "Typical folding mark (2.7 {})".format(metr_units), "col": "midnightblue"})
        title = "{} | traveled vs to_goal | {} | {} | {}".format(metric, full_cut, same, 'shrink' if shrink else 'unshrink')
        filename = "{}_traveled_vs_to_goal_{}_{}_{}".format(metric, full_cut, same, 'shrink' if shrink else 'unshrink')
        filename = os.path.join(custom_path, filename)
        fig_num = single_plot(fig_num, ax_prop, to_goal_arr, trav_arr,    legend_names.copy(), '.', 1,   bsf=False, rev=True,  extra_line=extra_line, shrink=shrink, xlab="Distance to the goal, {}".format(metr_units), ylab="Past dist, {}".format(metr_units), title=title, filename=filename)  # traveled vs to_goal | cut

        for i in range(len(to_goal_arr)):
            ff = legend_names[i].split('with')[1].split('ff')[0].strip()
            title = "{} | traveled vs to_goal | {} | {} | {} | {}".format(metric, full_cut, same, 'shrink' if shrink else 'unshrink', ff)
            filename = "{}_traveled_vs_to_goal_{}_{}_{}_{}".format(metric, full_cut, same, 'shrink' if shrink else 'unshrink', ff)
            filename = os.path.join(custom_path, filename)
            extra_line[1]["val"] = min(to_goal_arr[i])
            if metr_units == 'contacts':
                extra_line[1]["name"] = "The lowest {} metric ({} {})".format(metric.upper(), int(min(to_goal_arr[i])), metr_units)
            else:
                extra_line[1]["name"] = "The lowest {} metric ({:3.2f} {})".format(metric.upper(), min(to_goal_arr[i]), metr_units)
            fig_num = single_plot(fig_num, ax_prop, [to_goal_arr[i],], [trav_arr[i],], [legend_names[i],].copy(), '.', 1, bsf=False, rev=True, extra_line=extra_line, shrink=shrink,
                                  xlab="Distance to the goal, {}".format(metr_units), ylab="Past dist, {}".format(metr_units), title=title, filename=filename)  # traveled vs to_goal | cut

    if not shrink:
        for i in range(len(non_shrink_arr)):
            non_shrink_arr[i].insert(0, 0)
    ax_prop = {"min_lim_x": -max_len / 80, "max_lim_x": max_len + max_len / 80, "min_lim_y": 0, "max_lim_y": init_metr + init_metr / 80,  # max_non_init_rmsd + max_non_init_rmsd / 80,
               "min_ax_x": 0, "max_ax_x": max_len + max_len / 80, "min_ax_y": 0, "max_ax_y": init_metr + init_metr / 80, "ax_step_x": math.floor(max_len / 16), "ax_step_y": init_metr / 20}
    if metr_units == 'contacts':
        extra_line = [
            {"ax_type": 'hor', "val": init_metr, "name": "Initial {} metric ({} {})".format(metric.upper(), int(init_metr), metr_units), "col": "darkmagenta"},
            {"ax_type": 'hor', "val": min(min(elem) for elem in bsf_arr), "name": "The lowest {} metric ({} {})".format(metric.upper(), int(min(min(elem) for elem in bsf_arr)), metr_units), "col": "darkgreen"}]
    else:
        extra_line = [
            {"ax_type": 'hor', "val": init_metr, "name": "Initial {} metric ({:3.2f} {})".format(metric.upper(), init_metr, metr_units), "col": "darkmagenta"},
            {"ax_type": 'hor', "val": min(min(elem) for elem in bsf_arr), "name": "The lowest {} metric ({:3.2f} {})".format(metric.upper(), min(min(elem) for elem in bsf_arr), metr_units), "col": "darkgreen"}]
    if metric == 'rmsd':
        extra_line.append({"ax_type": 'hor', "val": 2.7, "name": "Typical folding mark (2.7 {})".format(metr_units), "col": "midnightblue"})
    title = "{} | to goal vs best_so_far | {} | {} | {}".format(metric, full_cut, same, 'shrink' if shrink else 'unshrink')
    filename = "{}_to_goal_vs_best_so_far_{}_{}_{}".format(metric, full_cut, same, 'shrink' if shrink else 'unshrink')
    filename = os.path.join(custom_path, filename)
    fig_num = single_plot(fig_num, ax_prop, bsf_arr, non_shrink_arr, legend_names.copy(), '-', 1, bsf=True, rev=False, extra_line=extra_line, shrink=shrink, xlab="Steps (20ps each)", ylab="Distance to the goal, {}".format(metr_units), title=title, filename=filename)  # to goal vs best_so_far | cut
    for i in range(len(bsf_arr)):
        ff = legend_names[i].split('with')[1].split('ff')[0].strip()
        title = "{} | to goal vs best_so_far | {} | {} | {} | {}".format(metric, full_cut, same, 'shrink' if shrink else 'unshrink', ff)
        filename = "{}_to_goal_vs_best_so_far_{}_{}_{}_{}".format(metric, full_cut, same, 'shrink' if shrink else 'unshrink', ff)
        extra_line[1]["val"] = min(bsf_arr[i])
        if metr_units == 'contacts':
            extra_line[1]["name"] = "The lowest {} metric ({} {})".format(metric.upper(), int(min(bsf_arr[i])), metr_units)
        else:
            extra_line[1]["name"] = "The lowest {} metric ({:3.2f} {})".format(metric.upper(), min(bsf_arr[i]), metr_units)
        filename = os.path.join(custom_path, filename)
        fig_num = single_plot(fig_num, ax_prop, [bsf_arr[i],], [non_shrink_arr[i],] if non_shrink_arr is not None else None, [legend_names[i],].copy(), '-', 1, bsf=True, rev=False, extra_line=extra_line, shrink=shrink, xlab="Steps (20ps each)",
                              ylab="Distance to the goal, {}".format(metr_units), title=title, filename=filename)  # to goal vs best_so_far | cut

    ax_prop = {"min_lim_x": max_non_init_rmsd, "max_lim_x": common_point-common_point/10, "min_lim_y": -max_len/80, "max_lim_y": max_len+max_len/80,
               "min_ax_x": common_point, "max_ax_x": max_non_init_rmsd, "min_ax_y": 0, "max_ax_y": max_len+max_len/80, "ax_step_x": (max_non_init_rmsd-common_point)/20, "ax_step_y": math.floor(max_len/20)}

    if metr_units == 'contacts':
        extra_line = [
            {"ax_type": 'ver', "val": init_metr, "name": "Initial {} metric ({} {})".format(metric.upper(), int(init_metr), metr_units), "col": "darkmagenta"},
            {"ax_type": 'ver', "val": min(min(elem) for elem in bsf_arr), "name": "The lowest {} metric ({} {})".format(metric.upper(), int(min(min(elem) for elem in bsf_arr)), metr_units), "col": "darkgreen"}]
    else:
        extra_line = [
            {"ax_type": 'ver', "val": init_metr, "name": "Initial {} metric ({:3.2f} {})".format(metric.upper(), init_metr, metr_units), "col": "darkmagenta"},
            {"ax_type": 'ver', "val": min(min(elem) for elem in bsf_arr), "name": "The lowest {} metric ({:3.2f} {})".format(metric.upper(), min(min(elem) for elem in bsf_arr), metr_units), "col": "darkgreen"}]
    if metric == 'rmsd':
        extra_line.append({"ax_type": 'hor', "val": 2.7, "name": "Typical folding mark (2.7 {})".format(metr_units), "col": "midnightblue"})
    title = "{} | best_so_far vs steps | {} | {} | {}".format(metric, full_cut, same, 'shrink' if shrink else 'unshrink')
    filename = "{}_best_so_far_vs_steps_{}_{}_{}".format(metric, full_cut, same, 'shrink' if shrink else 'unshrink')
    filename = os.path.join(custom_path, filename)
    fig_num = single_plot(fig_num, ax_prop, bsf_arr,     non_shrink_arr, legend_names.copy(), '-',  1,   bsf=True,  rev=True,  extra_line=extra_line, shrink=shrink, xlab="Distance to the goal, {}".format(metr_units), ylab="Steps (20 ps each)", title=title, filename=filename)  # best_so_far vs steps | cut
    for i in range(len(bsf_arr)):
        ff = legend_names[i].split('with')[1].split('ff')[0].strip()
        title = "{} | best_so_far vs steps | {} | {} | {} | {}".format(metric, full_cut, same, 'shrink' if shrink else 'unshrink', ff)
        filename = "{}_best_so_far_vs_steps_{}_{}_{}_{}".format(metric, full_cut, same, 'shrink' if shrink else 'unshrink', ff)
        extra_line[1]["val"] = min(bsf_arr[i])
        if metr_units == 'contacts':
            extra_line[1]["name"] = "The lowest {} metric ({} {})".format(metric.upper(), int(min(bsf_arr[i])), metr_units)
        else:
            extra_line[1]["name"] = "The lowest {} metric ({:3.2f} {})".format(metric.upper(), min(bsf_arr[i]), metr_units)
        filename = os.path.join(custom_path, filename)
        fig_num = single_plot(fig_num, ax_prop, [bsf_arr[i],], [non_shrink_arr[i],] if non_shrink_arr is not None else None, [legend_names[i],].copy(), '-', 1, bsf=True, rev=True, extra_line=extra_line, shrink=shrink,
                              xlab="Distance to the goal, {}".format(metr_units), ylab="Steps (20 ps each)", title=title, filename=filename)  # best_so_far vs steps | cut

    return fig_num


def single_plot(fig_num: int, ax_prop: dict, arr_A: list, arr_B: list, filenames_db: list, marker: str, mark_size: float,
                bsf: bool, rev: bool, shrink: bool, xlab: str, ylab: str, title: str, filename: str,
                extra_line: list = None, mdpi: int = 400, second_ax: dict = None, sec_arr: list = None) -> int:
    """Main plotting function

    Args:
        :param int fig_num: figure number, it should not matter, since we close all figures regularly
        :param dict ax_prop: axis properties
        :param list arr_A: typically Y values
        :param list arr_B: typically X values
        :param list filenames_db: line names
        :param str marker: type of the marker
        :param float mark_size: size of the marker
        :param bool bsf: best so far version
        :param bool rev: reversed
        :param bool shrink: whether to ignore x values, and just plot all y values
        :param str xlab: x label
        :param str ylab:  y label
        :param str title: plot title
        :param str filename: output filename
        :param list extra_line: whether to plot extra line, if so contains its properties
        :param int mdpi: plot resolution
        :param dict second_ax: whether to plot second Y axis, if so this contains dict with properties
        :param list sec_arr: value for the second axis

    Returns:
        :return: figure number, it should not matter, since we close all figures regularly
    """
    fig_num += 1
    # for fname in ['angl_version_of_best_traj_angl_only_results_gromos_trp_300_2_fixed_vs_pt_energy',
    #  'rmsd_version_of_best_traj_rmsd_only_results_gromos_trp_300_2_fixed_vs_pt_energy',
    #  'rmsd_version_of_best_traj_rmsd_vs_dist',
    #  'xor_version_of_best_traj_rmsd_only_results_opls_trp_300_2_fixed_vs_angl',
    #  'xor_version_of_best_traj_rmsd_only_results_opls_trp_300_2_fixed_vs_pt_energy',
    #  'xor_version_of_best_traj_angl_only_results_opls_trp_300_2_fixed_vs_pt_energy',
    #               'rmsd_to_goal_vs_best_so_far_full_RMSD_unshrink']:
    #     if fname in filename:
    #         print('found')

    w, h = figaspect(0.5)
    fig = plt.figure(fig_num, figsize=(w, h))
    #
    ax = fig.gca()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(w, h), sharex=True, squeeze=False)
    plt.xlim(ax_prop["min_lim_x"], ax_prop["max_lim_x"])
    plt.ylim(ax_prop["min_lim_y"], ax_prop["max_lim_y"])

    major_xticks = np.arange(ax_prop["min_ax_x"], ax_prop["max_ax_x"], ax_prop["ax_step_x"])
    major_yticks = np.arange(ax_prop["min_ax_y"], ax_prop["max_ax_y"], ax_prop["ax_step_y"])

    if ax_prop["ax_step_y"] is not None:
        if major_yticks[-1] > ax_prop["max_lim_y"]:  # fix inconsistency in real numbers
            major_yticks[-1] = ax_prop["max_lim_y"]
        if ax_prop["max_lim_y"] - major_yticks[-1] > ax_prop["ax_step_y"]:  # this should not happen, but just in case..
            major_yticks = np.append(major_yticks, major_yticks[-1] + ax_prop["ax_step_y"])
        elif ax_prop["max_lim_y"] - major_yticks[-1] > 0.7*ax_prop["ax_step_y"]:
            major_yticks = np.append(major_yticks, ax_prop["max_lim_y"])

    if ax_prop["ax_step_x"] is not None:
        if ax_prop["max_lim_x"] - major_xticks[-1] > ax_prop["ax_step_x"]:  # this should not happen, but just in case..
            print('2', filename)
            major_xticks = np.append(major_xticks, int(major_xticks[-1] + ax_prop["ax_step_x"]) if isinstance(ax_prop["ax_step_x"], int) else (major_xticks[-1] + ax_prop["ax_step_x"]))
        elif ax_prop["max_lim_x"] - major_xticks[-1] > 0.7 * ax_prop["ax_step_x"]:
            print('1', filename)
            major_xticks = np.append(major_xticks, int(ax_prop["max_lim_x"]) if isinstance(ax_prop["ax_step_x"], int) else ax_prop["max_lim_x"])

        if arr_B is not None and abs(arr_B[0][-1] - major_xticks[-1]) < 0.5 * ax_prop["ax_step_x"]:
            major_xticks[-1] = arr_B[0][-1]
        elif abs(max(len(elem) for elem in arr_A) - major_xticks[-1]) < 0.5 * ax_prop["ax_step_x"]:
            major_xticks[-1] = max(len(elem) for elem in arr_A)

    if major_xticks is not None:
        ax[0][0].set_xticks(major_xticks)
    if major_yticks is not None:
        ax[0][0].set_yticks(major_yticks)
    # if minor_xticks is not None:
    #     ax.set_xticks(minor_xticks, minor=True)
    # if minor_yticks is not None:
    #     ax.set_yticks(minor_yticks, minor=True)
    top_ax = ax[0][0]
    if second_ax is not None:
        ax2 = ax[0][0].twinx()
        major_yticks2 = np.arange(second_ax["min_ax_y"], second_ax["max_ax_y"], second_ax["ax_step_y"])

        if major_yticks2[-1] > second_ax["max_lim_y"]:  # fix inconsistency in real numbers
            major_yticks2[-1] = second_ax["max_lim_y"]

        if second_ax["max_lim_y"] - major_yticks2[-1] > second_ax["ax_step_y"]:
            major_yticks2 = np.append(major_yticks2, major_yticks2[-1] + second_ax["ax_step_y"])
        elif second_ax["max_lim_y"] - major_yticks2[-1] > 0.7*second_ax["ax_step_y"]:
            major_yticks2 = np.append(major_yticks2, second_ax["max_lim_y"])

        ax2.set_yticks(major_yticks2)
        ax2.tick_params(direction='out', length=6, width=1, grid_alpha=0.5)
        # ax[0].right_ax.set_ylim(second_ax["min_lim_y"], second_ax["max_lim_y"])
        ax2.set_ylim(second_ax["min_lim_y"], second_ax["max_lim_y"])
        ax2.plot(range(len(sec_arr)), sec_arr, color='r', alpha=0.75)
        ax2.set_ylabel(second_ax["label"] if second_ax["label"][-2] != ',' else second_ax["label"][0:-2])
        top_ax = ax2



    ax[0][0].tick_params(direction='out', length=6, width=1, grid_alpha=0.5)
    ax[0][0].grid(which='both', linestyle='dotted')
    plt.xticks(rotation=30)
    plt.subplots_adjust(top=0.95, bottom=0.16, left=0.09, right=0.90)

    lines_b = []
    for i, bsf_trav_to_goal in enumerate(arr_A):
        if not shrink:  # use provided array arr_B
            if rev:
                line_b, = ax[0][0].plot(arr_A[i], arr_B[i], marker, markersize=mark_size, alpha=0.75)
            else:
                line_b, = ax[0][0].plot(arr_B[i], arr_A[i], marker, markersize=mark_size, alpha=0.75)
        else:  # generate array from 0 to len(arr_A)
            if rev:
                if bsf:
                    line_b, = ax[0][0].plot(arr_A[i], range(len(arr_A[i])), marker, markersize=mark_size, alpha=0.75)
                else:
                    line_b, = ax[0][0].plot(arr_A[i], arr_B[i], marker, markersize=mark_size, alpha=0.75)
            else:
                line_b, = ax[0][0].plot(range(len(arr_A[i])), arr_A[i], marker, markersize=mark_size, alpha=0.75)
        lines_b.append(line_b)

    if extra_line is not None:
        for el in extra_line:
            if el["ax_type"] == 'ver':
                straight_line = ax[0][0].axvline(x=el["val"], color=el["col"], linestyle='--', alpha=0.75)  #
            elif el["ax_type"] == 'hor':
                straight_line = ax[0][0].axhline(y=el["val"], color=el["col"], linestyle='--', alpha=0.75)
            else:
                raise Exception('Wrong ax type')
            lines_b.append(straight_line)
            filenames_db.append(el["name"])
        if el["ax_type"] == 'ver':
            if not rev:
                ax[0][0].annotate('Folding direction', xytext=(ax_prop["min_ax_x"] + 1 * ax_prop["ax_step_x"], ax_prop["max_lim_y"] - 1 * ax_prop["ax_step_y"]), xy=(ax_prop["min_ax_x"] + 5 * ax_prop["ax_step_x"], ax_prop["max_lim_y"] - 1 * ax_prop["ax_step_y"]), arrowprops={'arrowstyle': '->', 'lw': 1.3, 'color': 'mediumblue'}, va='center')  # -->
            else:
                ax[0][0].annotate('Folding direction', xytext=(ax_prop["max_ax_x"] - 1 * ax_prop["ax_step_x"], ax_prop["max_lim_y"] - 1 * ax_prop["ax_step_y"]), xy=(ax_prop["max_ax_x"] - 5 * ax_prop["ax_step_x"], ax_prop["max_lim_y"] - 1 * ax_prop["ax_step_y"]), arrowprops={'arrowstyle': '->', 'lw': 1.3, 'color': 'mediumblue'}, va='center')  # -->
        else:
            if not rev:
                if second_ax is not None:
                    ax2.annotate('Folding direction', xytext=(ax_prop["min_ax_x"] + 3.5 * ax_prop["ax_step_x"], second_ax["max_lim_y"] - 1 * second_ax["ax_step_y"]), xy=(ax_prop["min_ax_x"] + 3.5 * ax_prop["ax_step_x"], second_ax["max_lim_y"] - 4 * second_ax["ax_step_y"]), arrowprops={'arrowstyle': '->', 'lw': 1.3, 'color': 'mediumblue'}, ha='center')  # <--
                else:
                    ax[0][0].annotate('Folding direction', xytext=(ax_prop["min_ax_x"] + 3.5 * ax_prop["ax_step_x"], ax_prop["max_lim_y"] - 1 * ax_prop["ax_step_y"]), xy=(ax_prop["min_ax_x"] + 3.5 * ax_prop["ax_step_x"], ax_prop["max_lim_y"] - 4 * ax_prop["ax_step_y"]), arrowprops={'arrowstyle': '->', 'lw': 1.3, 'color': 'mediumblue'}, ha='center')  # <--
            else:
                pass # does not exist
                # ax.annotate('folding direction', xytext=(ax_prop["min_ax_x"] + 1 * ax_prop["ax_step_x"], ax_prop["max_lim_y"] - 1 * ax_prop["ax_step_y"]), xy=(ax_prop["min_ax_x"] + 1 * ax_prop["ax_step_x"], ax_prop["max_lim_y"] - 4 * ax_prop["ax_step_y"]), arrowprops={'arrowstyle': '->', 'lw': 1.5, 'color': 'mediumblue'}, ha='center')  # -->

    if second_ax is not None:
        lines_b.append(ax[0][0].plot([], [], marker, color='r', markersize=mark_size)[0])
        filenames_db.append(second_ax["line_name"])

    ax[0][0].set_xlabel(xlab)
    ax[0][0].set_ylabel(ylab if ylab[-2] != ',' else ylab[0:-2])
    top_ax.legend(lines_b, filenames_db)
    plt.title(title)
    try:
        plt.savefig(filename, dpi=mdpi, transparent=True, bbox_inches='tight', pad_inches=0.02)
    except:
        plt.show()
    plt.close('all')
    return fig_num


if __name__ == '__main__':
    main()
