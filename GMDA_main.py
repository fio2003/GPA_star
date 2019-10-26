#!/usr/bin/env python3

"""
This file contains main computational loop and functions highly related to it
.. module:: GMDA_main
    :platform: linux

.. moduleauthor:: Ivan Syzonenko <is2k@mtmail.mtsu.edu>
"""
__license__ = "MIT"
__docformat__ = 'reStructuredText'

import heapq
import time
import os
import multiprocessing as mp
import numpy as np
from shutil import copy2 as cp2
from pathlib import Path
import zlib
import gc

from typing import NoReturn

from db_proc import insert_into_log, insert_into_main_stor, insert_into_visited, copy_old_db
from helper_funcs import trjcat_many, make_a_step, create_core_mapping, get_seed_dirs, check_precomputed_noize, \
                         get_new_seeds, get_digest, make_a_step2, rm_seed_dirs, make_a_step3, main_state_recover, supp_state_recover, \
                         main_state_backup, supp_state_backup
from parse_topology_for_hydrogens import parse_top_for_h
from gmx_wrappers import gmx_trjcat, gmx_trjconv
from metric_funcs import get_knn_dist_mdsctk, get_bb_to_angle_mdsctk, get_angle_to_sincos_mdsctk, \
                         get_native_contacts, gen_file_for_amb_noize, save_an_file, compute_init_metric, compute_metric, \
                         select_metrics_by_snr, get_contat_profile_mdsctk
# from pympler import muppy, summary
# from memory_profiler import profile
# import sys
MAX_ITEMS_TO_HANDLE = 50000
# extra_past = './'  # define extra past dir - this is temporary handle.

# def proc_local_minim(open_queue, best_so_far_name: str, tol_error, ndx_file: str, name_2_digest_map: dict, goal_top: str, local_minim_names: list):
#     """
#     Deprecated approach to block falling into local minima basin
#     :param open_queue: sorted queue that contains nodes about to be processed. This is actually only a partial queue (only top elements)
#     :param best_so_far_name: name of the trajectory closest to the goal (according to the current metric)
#     :param tol_error: minimal metric vibration of the NMR structure
#     :param ndx_file: .ndx - index of the protein atoms of the current conformation
#     :param name_2_digest_map: dictionary that maps trajectory name to it's precomputed digest
#     :param goal_top: .top - topology of the NMR conformation
#     :param local_minim_names: list of nodes close to the local minima
#     :return:
#     """
#     # split name into subnames
#     # compute distance
#     # import math
#     range_lim = 6
#     strict = True
#     if strict:
#         basin_err = tol_error * 4
#         stem_err = lambda i: tol_error - 2 * tol_error * i / 5
#     else:
#         basin_err = tol_error * 2
#         stem_err = lambda i: tol_error - tol_error * i / 5
#
#     prev_points = best_so_far_name.split('_')
#     past_dir = './past'
#     # len_prev_points = len(prev_points)
#     # step = len_prev_points//18
#     all_prev_names = ['_'.join(prev_points[:i]) for i in range(1, len(prev_points))]
#     hashed_names = [os.path.join(past_dir, name_2_digest_map[point] + '.xtc') for point in all_prev_names]
#     len_hashed_names = len(hashed_names)
#     closest_to_minim = hashed_names[len_hashed_names:len_hashed_names // 2:-1]
#     gmx_trjcat(f=closest_to_minim, o='local_min.xtc', n=ndx_file, cat=True, vel=False, sort=False, overwrite=True)
#     # range_lim = min(6, len_prev_points)
#
#     hashed_names = [name_2_digest_map[name[4]] for name in open_queue]
#
#     trjcat_many(hashed_names, past_dir, './combinded_traj_openq.xtc')
#
#     rmsd = get_knn_dist_mdsctk('./combinded_traj_openq.xtc', 'local_min.xtc', goal_top)
#
#     rmsd_structured = list()
#     for i in range(len(closest_to_minim)):
#         rmsd_structured.append(rmsd[i * len(hashed_names):(i + 1) * len(hashed_names)])
#
#     # next part of code implements gradual pruning:
#     # the closer point to the end of perfect path - the closer we are to the local minim center
#     # so we need to remove all near points.
#     # some extra code here to handle case when we have shorter paths and make sure that
#     # the most pruning will receive only center
#     step = len(rmsd_structured)//range_lim if len(rmsd_structured) > range_lim else 1
#     how_many = [0]
#     sum = 0
#     for i in range(1, range_lim):
#         sum += step
#         how_many.append(sum)
#         if sum == len(rmsd_structured):
#             break
#     how_many[-1] += len(rmsd_structured) - step * (len(how_many) - 1)
#     set_of_points_to_remove = set()
#
#     for i in range(len(how_many)-1):
#         subarr = rmsd_structured[how_many[i]:how_many[i+1]]
#         for line_of_points in subarr:
#             for point_pos, point in enumerate(line_of_points):
#                 if point < stem_err(i):
#                     set_of_points_to_remove.add(point_pos)
#
#     print('Main stem, trimming {} points'.format(len(set_of_points_to_remove)))
#
#     # at this point we cleaned main stem of perfect path
#     # now its time to clean local minimum basin
#
#     hashed_names = [name_2_digest_map[name] for name in local_minim_names]
#     trjcat_many(hashed_names, past_dir, './combinded_traj_basin.xtc')
#
#     if os.path.exists('./local_minim_bas.xtc'):
#         gmx_trjcat(f=['./combinded_traj_basin.xtc', 'local_minim_bas.xtc'],
#                    o='./combinded_traj_basin_comb.xtc',
#                    n='./prot_dir/prot.ndx', cat=True, vel=False, sort=False, overwrite=True)
#         os.remove('./combinded_traj_basin.xtc')
#         os.rename('./combinded_traj_basin_comb.xtc', './combinded_traj_basin.xtc')
#
#     gmx_trjcat(f=['./combinded_traj_basin.xtc', 'local_min.xtc'],
#                o='./local_minim_bas.xtc',
#                n='./prot_dir/prot.ndx', cat=True, vel=False, sort=False, overwrite=True)
#
#     rmsd = get_knn_dist_mdsctk('./combinded_traj_openq.xtc', './combinded_traj_basin.xtc', goal_top)
#
#     rmsd_structured = list()
#     for i in range(len(closest_to_minim)):
#         rmsd_structured.append(rmsd[i * len(hashed_names):(i + 1) * len(hashed_names)])
#
#     for line_of_points in rmsd_structured:
#         for point_pos, point in enumerate(line_of_points):
#             if point < basin_err:
#                 set_of_points_to_remove.add(point_pos)
#
#     print('Total points to trim: {} points'.format(len(set_of_points_to_remove)))
#
#     open_queue = [node for index, node in enumerate(open_queue) if index not in set_of_points_to_remove]
#     # heapq.heappush(open_queue, elem)
#     heapq.heapify(open_queue)
#     return open_queue


# def check_local_minimum(temp_xtc_file: str, goal_top: str, tol_error: float):
#     """
#     Checks whether tested frames are close to the local minima basin
#     :param temp_xtc_file: frames to check
#     :param goal_top: .top - topology of the NMR conformation
#     :param tol_error: minimal metric vibration of the NMR structure
#     :return: True if belongs, False otherwise
#     """
#     if os.path.exists('./local_minim_bas.xtc'):
#         strict = True
#         if strict:
#             prune_err = tol_error*4
#         else:
#             prune_err = tol_error * 2
#         min_dist = min(get_knn_dist_mdsctk(temp_xtc_file, 'local_minim_bas.xtc', goal_top))
#         if min_dist < prune_err:
#             return False
#     return True


def queue_rebuild(process_queue: list, open_queue_to_rebuild: list, node_info: dict, cur_mult: float, new_metr_name: str, sep_proc: bool = True) -> list:
    """Resorts the queue according to the new metric.

    Args:
        :param list process_queue: queue to use if function is executed in a separate process
        :param list open_queue_to_rebuild: sorted queue that contains nodes about to be processed. This is actually only a partial queue (only top elements)
        :param dict node_info:
        :param float cur_mult: current greedy factor
        :param str new_metr_name: defines how to sort the new queue
        :param bool sep_proc: whether the function runs in a separate process

    Returns:
        :return: if separate process - then new queue and metric name are pushed into the queue, otherwise returned
        :rtype: list
    """
    gc.collect()
    new_queue = list()
    to_goal, total = '{}_to_goal'.format(new_metr_name), '{}_dist_total'.format(new_metr_name)
    try:
        for elem in open_queue_to_rebuild[1:]:
            heapq.heappush(new_queue, (cur_mult*node_info[elem[2]][total] + node_info[elem[2]][to_goal], 0, elem[2]))
    except Exception:
        print(len(node_info))
        print(len(open_queue_to_rebuild))
        print(new_metr_name)
        print(cur_mult)
        print(sep_proc)
    del open_queue_to_rebuild
    gc.collect()
    if sep_proc:
        process_queue.put((new_queue, new_metr_name))
    else:
        return new_queue


def get_atom_num(ndx_file: str) -> int:
    """Computes number of atoms in the particular index file.

    Args:
        :param str ndx_file: .ndx - index of the protein atoms of the current conformation.

    Returns:
        :return: number of atoms in the .ndx file.
        :rtype: int
    """
    with open(ndx_file, 'r') as index_file:
        index_file.readline()  # first line is the comment - skip it
        indices = index_file.read().strip()
    elems = indices.split()
    atom_num = len(elems)
    return atom_num


def parse_hostnames(seednum: int, hostfile: str = 'hostfile') -> tuple:
    """Spreads the load among the hosts found in the hostfile. Needed for MPI

    Args:
        :param seednum: total number of seeds used in the current run
        :param hostfile: filename of the hostfile

    Returns:
        :return: hosts split partitioned according to the number of seeds and total number of cores for each job
    """
    with open(hostfile, 'r') as f:
        hosts = f.readlines()
    del hostfile
    hostnames = [elem.strip().split(' ')[0] for elem in hosts]
    ncores = [int(elem.strip().split(' ')[1].split('=')[1]) for elem in hosts]
    ev_num = len(hosts) // seednum
    if ev_num == 0:
        raise Exception('Special case is not implemented')
    else:
        chopped = [tuple(hostnames[i:i+ev_num]) for i in range(0, len(hostnames), ev_num)]
        ncores_sum = [sum(ncores[i:i+ev_num]) for i in range(0, len(ncores), ev_num)]
    return chopped, ncores_sum


def compute_on_local_machine(cpu_map: list, seed_list: list, cur_name: str, past_dir: str, work_dir: str, seed_dirs: dict,
                             topol_file_init: str, ndx_file_init: str, old_name_digest: str) -> tuple:
    """This version is optimised for usage on one machine with tMPI (see GROMACS docs).

    Performs check whether requested simulation was completed in the past.
    If so (and all requested files exist), we skip the computation,
    otherwise we start the sequence of events that prepare and run the simulation in the separate process.
    I was playing with better core distribution, but it did not work well, since GROMACS may complain when you assign odd number of cores, or when 14 cores does not work, but 12 and 16 are fine.
    What I know fo sure that powers of 2 work the best until 128 cores, but we do not have so many cores on one machine.
    Two machines are worse than one (yes, 64+64 is slower than 64, same with 32+32) - maybe Infiniband can help, but we do not have one.
    Additionally, I commented prev_runs - it just uses more RAM without giving any significant speedup.

    Args:
        :param list cpu_map: number of cores for particular task (seed)
        :param list seed_list: list of current seeds
        :param str cur_name: name of the current node (prior path constructed from seed names s_0_1_4)
        :param str past_dir: path to the directory with prior computations
        :param str work_dir: path to the directory where seed dirs reside
        :param dict seed_dirs: dict which contains physical path to the directory where simulation with particular seed is performed
        :param str topol_file_init: .top - topology of the initial (unfolded) conformation
        :param str ndx_file_init: .ndx - index of the protein atoms of the unfolded conformation
        :param list prev_runs_files: information about all previously generated files in ./past directory
        :param str old_name_digest: digest of the current name

    Returns:
        :return: array of PIDs to join them later and allow some more parallel computation, hash names, simulation names.
        :rtype: tuple

    Returns: PIDs and new filenames. PIDs - to join processes later.
    """
    files_for_trjcat = list()
    recent_filenames = list()
    pid_arr = list()
    # recent_n2d = dict()
    # recent_d2n = dict()
    for i, exec_group in enumerate(cpu_map):
        saved_cores = 0
        for cur_group_sched in exec_group:
            cores, seed_2_process = cur_group_sched
            seed_2_process = seed_list[seed_2_process]
            new_name = '{}_{}'.format(cur_name, seed_2_process)
            seed_digest_filename = get_digest(new_name)
            # recent_n2d[new_name] = seed_digest_filename
            # recent_d2n[seed_digest_filename] = new_name
            xtc_filename = '{}.xtc'.format(seed_digest_filename)
            gro_filename = '{}.gro'.format(seed_digest_filename)

            files_for_trjcat.append(os.path.join(past_dir, xtc_filename))
            # # if os.path.exists(os.path.join('./past', xtc_filename)) and os.path.exists(os.path.join('./past', gro_filename)):
            #     saved_cores += cores  # not fair, but short TODO: write better logic for cores remapping
            #     recent_filenames.append(xtc_filename)
            #     recent_filenames.append(gro_filename)
            #     continue
            # else:
            if not (os.path.exists(os.path.join(past_dir, xtc_filename)) and os.path.exists(os.path.join(past_dir, gro_filename))): #\
                # and not (os.path.exists(os.path.join(extra_past, xtc_filename)) and os.path.exists(os.path.join(extra_past, gro_filename))):
                md_process = None
                md_process = mp.Process(target=make_a_step,
                                        args=(work_dir, seed_2_process, seed_dirs, topol_file_init, ndx_file_init,
                                              seed_digest_filename, old_name_digest, past_dir, cores + saved_cores))
                md_process.start()
                # print('Process started :{} pid:{} alive:{} ecode:{} with next param: s:{}, pd:{}, cor:{}'.format(md_process.name,
                # md_process.pid, md_process.is_alive(), md_process.exitcode, seed_2_process, past_dir, cores+saved_cores))
                pid_arr.append(md_process)
                # make_a_step(work_dir, seed_2_process, seed_dirs, seed_list, topol_file, ndx_file, name_2_digest_map,
                # cur_job_name, past_dir, cores+saved_cores)
                saved_cores = 0
                # print('md_process{} '.format(seed_2_process), end="")
                # recent_filenames.append(xtc_filename)
                # recent_filenames.append(gro_filename)
        if i is not len(cpu_map) - 1:  # if it is not the last portion of threads then wait for completion
            [proc.join() for proc in pid_arr]

    # combine prev_step and goal to compute two dist in one pass
    # rm_queue.join()  # make sure that queue is empty (all files were deleted)

    # Test code for multiprocessing check. There was a problem with python3.4 and old sqlite (too many parallel
    # connections when reusing past results).
    # [proc.join(timeout=90) for proc in pid_arr]
    # if len(pid_arr):
    #     print('Proc arr is not empty:', end=' ')
    #     while True:
    #         proc_stil_running = 0
    #         for cur_group_sched in pid_arr:
    #             print('waiting for name:{} pid:{} alive:{} ecode:{}'.format(cur_group_sched.name,
    #             cur_group_sched.pid, cur_group_sched.is_alive(), cur_group_sched.exitcode))
    #             cur_group_sched.join(timeout=40)
    #             if cur_group_sched.exitcode is not None:
    #                 proc_stil_running += 1
    #         if proc_stil_running == len(pid_arr):
    #             print('Done.')
    #             break

    # if len(pid_arr):
    #     print('j{} '.format(len(pid_arr)), end="")
    return pid_arr, files_for_trjcat, recent_filenames, None, None  # recent_n2d, recent_d2n


def compute_with_mpi(seed_list: list, cur_name: str, past_dir: str, work_dir: str, seed_dirs: dict, topol_file_init: str,
                     ndx_file_init: str, old_name_digest: str, tot_seeds: int, hostnames: list,
                     ncores: list, sched: bool = False, ntomp: int = 1) -> tuple:
    """This version is optimised for usage on more than one machine with tMPI and/or MPI.

    If you use scheduler and know exactly how many cores each machine has - supply correct hostfile and use tMPI on each machine with OMP.
    If you use scheduler without option to choose specific machine - use version without scheduler or local version (depends on your cluster implementation).
    Performs check whether requested simulation was completed in the past.
    If so (and all requested files exist), we skip the computation,
    otherwise we start the sequence of events that prepare and run the simulation in the separate process.
    I was playing with better core distribution, but it did not work well, since GROMACS may complain when you assign odd number of cores, or when 14 cores does not work, but 12 and 16 are fine.
    What I know fo sure that powers of 2 work the best until 128 cores, but we do not have so many cores on one machine.
    Two machines are worse than one (yes, 64+64 is slower than 64, same with 32+32) - maybe InfiniBand can help, but we do not have one.
    Additionally, I commented prev_runs - it just uses more RAM without giving any significant speedup.

    Args:
        :param list seed_list: list of current seeds
        :param str cur_name: name of the current node (prior path constructed from seed names s_0_1_4)
        :param str past_dir: path to the directory with prior computations
        :param strwork_dir: path to the directory where seed dirs reside
        :param dict seed_dirs: dict which contains physical path to the directory where simulation with particular seed is performed
        :param str topol_file_init:  .top - topology of the initial (unfolded) conformation
        :param str ndx_file_init: .ndx - index of the protein atoms of the initial (unfolded) conformation
        :param list prev_runs_files: information about all previously generated files in ./past directory
        :param str old_name_digest: digest of the current name
        :param int tot_seeds: total number of seeds, controversial optimisation.
        :param list hostnames: correct names/IPs of the hosts
        :param int ncores: number of cores on each host
        :param bool sched: secelts proper make_a_step version
        :param int ntomp: how many OMP threads use during the MD simulation (2-4 is the optimal value on 32-64 core hosts)

    Returns:
        :return: array of PIDs to join them later and allow some more parallel computation, hash names, simulation names.
        :rtype: tuple

    PIDs and new filenames. PIDs - to join processes later.
    """
    # if os.path.exists(os.path.join(os.getcwd(), 'local.comp')):
    #     hostnames = [('Perseus', )]*tot_seeds
    gc.collect()
    files_for_trjcat = list()
    recent_filenames = list()
    pid_arr = list()
    # recent_n2d = dict()
    # recent_d2n = dict()
    for i in range(tot_seeds):
        seed_2_process = seed_list[i]
        new_name = '{}_{}'.format(cur_name, seed_2_process)
        seed_digest_filename = get_digest(new_name)
        # recent_n2d[new_name] = seed_digest_filename
        # recent_d2n[seed_digest_filename] = new_name
        xtc_filename = '{}.xtc'.format(seed_digest_filename)
        gro_filename = '{}.gro'.format(seed_digest_filename)

        # if os.path.exists(os.path.join(extra_past, xtc_filename)) and os.path.exists(os.path.join(extra_past, gro_filename)):
        #     files_for_trjcat.append(os.path.join(extra_past, xtc_filename))
        # else:
        files_for_trjcat.append(os.path.join(past_dir, xtc_filename))

        if not (os.path.exists(os.path.join(past_dir, xtc_filename)) and os.path.exists(os.path.join(past_dir, gro_filename))): # \
            # make_a_step2(work_dir, seed_2_process, seed_dirs, topol_file_init, ndx_file_init, seed_digest_filename, old_name_digest,
            # past_dir, hostnames[i], ncores[i])
            if sched:
                md_process = mp.Process(target=make_a_step3,
                                        args=(work_dir, seed_2_process, seed_dirs, topol_file_init, ndx_file_init,
                                              seed_digest_filename, old_name_digest, past_dir, int(ncores/tot_seeds), ntomp))
            else:
                md_process = mp.Process(target=make_a_step2,
                                        args=(work_dir, seed_2_process, seed_dirs, topol_file_init, ndx_file_init,
                                              seed_digest_filename, old_name_digest, past_dir, hostnames[i], ncores[i]))
            md_process.start()
            pid_arr.append(md_process)
        recent_filenames.append(xtc_filename)
        recent_filenames.append(gro_filename)

    return pid_arr, files_for_trjcat, recent_filenames, None, None  # recent_n2d, recent_d2n


def check_in_queue(queue: list, elem_hash: str) -> bool:
    """Checks whether elements with provided hash exists in the queue

    Args:
        :param list queue: specific queue to check
        :param str elem_hash: name to find in the queue

    Returns:
        :return: True if element found, False otherwise
        :rtype: bool
    """
    for elem in queue:
        if elem[2] == elem_hash:
            return True
    return False


def second_chance(open_queue: list, visited_queue: list, best_so_far_name: str, cur_metric: str, main_dict: dict,
                  node_max_att: int, cur_metric_name: str, best_so_far: dict, tol_error: float, greed_mult: float) -> list:
    """Typically executed during the seed change.

    We want to give the second chance to a promising trajectories with different seeds. Typically, we allow up to 4 attempts.
    However, the best trajectories are always readded to the queue.

    Args:
        :param list open_queue: sorted queue that contains nodes about to be processed. This is actually only a partial queue (only top elements)
        :param list visited_queue: sorted queue that contains nodes  processed prior. This is actually only a partial queue (only top elements)
        :param str best_so_far_name: node with the closest distance to the goal according to
        the guiding metric - we want to keep it for a long time, with hope that it will jump over the energy barrier
        :param str cur_metric: index of the current metric
        :param dict main_dict: map with all the information (prior and goal distances for all metrics, names, hashnames, attempts, etc)
        :param int node_max_att: defines how many attempts each node can have
        :param str cur_metric_name: name of the current metric
        :param dict best_so_far: name of the node with the closest metric distance to the goal
        :param float tol_error: minimal metric vibration of the NMR structure
        :param float greed_mult: greedy multiplier, used to assign correct metric value (ballance between optimality and greedyness)

    Returns:
        :return: short list of promising nodes, they will be merged with the open queue later
        :rtype: list
    """

    res_arr = list()
    recover_best = True
    for elem in open_queue:
        if elem[2] == best_so_far_name[cur_metric_name]:
            recover_best = False
            break

    for elem in visited_queue:  # elem structure: tot_dist, att, cur_name
        # we give node_max_att attempts for a node to make progress with different seed
        if (elem[1] < node_max_att and main_dict[elem[2]]['{}_to_goal'.format(cur_metric_name)] - best_so_far[cur_metric_name] < tol_error[cur_metric_name]):  # \
            # and elem[2] != best_so_far_name[cur_metric]:
            # or main_dict[elem[2]]['{}_to_goal'.format(cur_metric_name)] != best_so_far[cur_metric]:
            if elem[2] == best_so_far_name[cur_metric_name]:
                if recover_best:
                    res_arr.append(elem)
                    recover_best = False
                    break
            else:
                if elem[1] > 1 and check_in_queue(open_queue, elem[2]):
                    print('Not adding regular node (already in the queue)')
                else:
                    res_arr.append(elem)
                    print('Readding "{}" with attempt counter: {} and dist: {}'.format(elem[2], elem[1], elem[0]))

    elem = main_dict[best_so_far_name[cur_metric_name]]
    if recover_best:
        res_arr.append((elem['{}_dist_total'.format(cur_metric_name)] * greed_mult + elem['{}_to_goal'.format(cur_metric_name)],
                        0, best_so_far_name[cur_metric_name]))
        print('Recovering best')
    else:
        print('Not recovering best (already in the open queue)')
    del elem

    return res_arr


def check_dupl(name_to_check: str, visited_queue: list) -> list:
    """
    This function is just a detector of duplicates.

    Main source of dupplicates is when the algorithme gives the second chance to the same seed, but does not use it.
    This function checks whether specific name was used recently

    Args:
        :param name_to_check: name that is about to be sampled
        :param visited_queue: all previously used names

    Returns:
        :return: True if name was used recently, otherwise False
    """
    arr = [name[2] for name in visited_queue]
    if name_to_check in arr:
        print("Duplicate found in {} last elements, index: {}\nelem:{}".format(len(arr), arr.index(name_to_check), name_to_check))
        return True
    return False


def define_rules() -> list:
    """Generates rules to make metric usage more flexible thus reduce unproductive CPU cycles.

    Rules are generated according to the next scheme:
        rule: [rule_num {num or None}] [condition] [action]
        condition : [metr_val/iter] [value] [metr_name] [lower/higher/equal]
        action: [put/remove/switch] [metr_name]
        @ - indicates initial metr value
        Example:
        [0], [metr_val 0.7@ AARMSD lower], [switch BBRMSD]
        [1], [metr_val 0.5@ BBRMSD lower], [put ANGL]
        [2], [metr_val 0.4@ BBRMSD lower], [put AND_H]
        [3], [metr_val 0.7 BBRMSD lower],  [remove BBRMSD]

    Returns:
        :return: all defined rules in a sorted order.
        :rtype: list.
    """

    metric_rules = list()
    #                    #         condition                               action
    metric_rules.append((0, ["metr_val", "0.7@", "AARMSD", "lower"], ["switch", "BBRMSD"]))
    metric_rules.append((1, ["metr_val", "7", "BBRMSD", "lower"],    ["remove", "AARMSD"]))
    metric_rules.append((2, ["metr_val", "7", "BBRMSD", "lower"],    ["put", "ANGL"]))
    metric_rules.append((3, ["metr_val", "3.5", "BBRMSD", "lower"],    ["put", "AARMSD"]))
    metric_rules.append((4, ["metr_val", "3", "BBRMSD", "lower"],    ["put", "AND_H"]))
    metric_rules.append((5, ["metr_val", "2.5", "AARMSD", "lower"],    ["put", "AND"]))
    metric_rules.append((6, ["metr_val", "2.5", "AARMSD", "lower"],    ["put", "XOR"]))

    return metric_rules


def check_rules(metrics_sequence: list, rules: list, best_so_far: dict, init_metr: dict, metric_names: list, cur_gc: int) -> tuple:
    """Checks custom conditions and adds/removes available metrics.

    For each rule, we check the condition.
    If it is true - we apply the action and remove the rule.

    Args:
        :param list metrics_sequence: currently available metrics
        :param list rules: current list of rules
        :param dict best_so_far: lowest distance to the goal for each metric
        :param dict init_metr: initial distance to the goal for each metric
        :param list metric_names: list of all metrics to check proper metric name in the rule
        :param int cur_gc: gurrent value of the greedy_counter since

    Returns:
        :return: updated list of rules, updated list of alowed metrics,
        and metric to switch if appropriate rule was activated.
        :rtype: tuple
    """
    switch_metric = None
    rules_to_remove = list()
    for rule in rules:
        perform_action = False
        condition = rule[1]
        if condition[0] == 'metr_val':
            cond_metr = condition[2]
            compar_val = float(condition[1]) if '@' not in condition[1] else float(condition[1][:-1])*init_metr[cond_metr]
            if condition[3] == 'lower' and best_so_far[cond_metr] < compar_val:
                perform_action = True
            elif condition[3] == 'higher' and best_so_far[cond_metr] > compar_val:
                perform_action = True
            elif condition[3] == 'equal' and best_so_far[cond_metr] == compar_val:
                perform_action = True
            else:
                continue
        else:
            # this is where you need exact cur_gc, so you still can check
            raise Exception("Not implemented")

        if perform_action:
            action = rule[2]
            if action[0] == 'put' and action[1] in metric_names and action[1] not in metrics_sequence:
                metrics_sequence.append(action[1])
            if action[0] == 'remove' and action[1] in metrics_sequence:
                metrics_sequence.remove(action[1])
            if action[0] == 'switch' and action[1] in metric_names:
                if cur_gc >= 120:
                    continue
                switch_metric = action[1]
                if action[1] not in metrics_sequence:
                    print('You were trying to switch to {}, but it was not in the list of metrics.\nAdding it to the list.\n')
                    metrics_sequence.append(action[1])
            rules_to_remove.append(rule[0])
    if len(rules_to_remove):
        rules = [rule for rule in rules if rule[0] not in rules_to_remove]

    return rules, metrics_sequence, switch_metric


# def GMDA_main(prev_runs_files: list, past_dir: str, print_queue: mp.JoinableQueue,
#               db_input_queue: mp.JoinableQueue, copy_queue: mp.JoinableQueue, rm_queue: mp.JoinableQueue, tot_seeds: int = 4) -> NoReturn:
def GMDA_main(past_dir: str, print_queue: mp.JoinableQueue, db_input_queue: mp.JoinableQueue, tot_seeds: int = 4) -> NoReturn:
    """This is the main loop.

    Note that it has many garbage collector calls - it can slightly reduce the performance, but also reduces total memory usage.
    Feel free to comment them - they do not affect the algorithm

    Args:
        :param list prev_runs_files you may see this as the list of files found before the execution.
         We do not use it anymore to reduce the memory footprint.
         Instead we check existence of the file separately.
        :param str past_dir: location of all generated .gro, .xtc, metric values. Sequence of past seeds results in the unique name.
        :type past_dir: str
        :param mp.JoinableQueue print_queue: separate thread for printing operations, connected to the main process by Queue.
         It helps significantly during the restart without the previously saved state:
         you can query DB faster without waiting for printing operations to complete.
        :param mp.JoinableQueue db_input_queue:
        :param mp.JoinableQueue copy_queue: connection to the separate process that handled async copy. Should be rewriten with asyncio
        :param mp.JoinableQueue rm_queue: connection to the separate process that handled async rm. Should be rewriten with asyncio
        :param int tot_seeds: number of parallel seeds to be executed - very powerful knob

    Returns:
        :return: Nothing, once stop condition is reached, looping stops and returns to the parent to join/clean other threads
    """

    possible_prot_states = ['Full_box', 'Prot', 'Backbone']
    print('Main process rebuild_queue_process: ', os.getpid())
    gc.collect()
    prot_dir = os.path.join(os.getcwd(), 'prot_dir')
    if not os.path.exists(prot_dir):
        os.makedirs(prot_dir)
    print('Prot dir: ', prot_dir)
    # These files has to be in prot_dir
    init = os.path.join(prot_dir, 'init.gro')  # initial state, will be copied into work dir, used for MD
    goal = os.path.join(prot_dir, 'goal.gro')  # final state, will not be used, but needed for derivation of other files

    topol_file_init = os.path.join(prot_dir, 'topol_unfolded.top')  # needed for MD
    topol_file_goal = os.path.join(prot_dir, 'topol_folded.top')  # needed for MD

    ndx_file_init = os.path.join(prot_dir, 'prot_unfolded.ndx')  # needed for extraction of protein data
    ndx_file_goal = os.path.join(prot_dir, 'prot_folded.ndx')  # needed for extraction of protein data

    init_bb_ndx = os.path.join(prot_dir, 'bb_unfolded.ndx')
    goal_bb_ndx = os.path.join(prot_dir, 'bb_folded.ndx')

    # These files will be generated
    init_xtc = os.path.join(prot_dir, 'init.xtc')  # small version, used for rmsd
    init_xtc_bb = os.path.join(prot_dir, 'init_bb.xtc')  # small version, used for rmsd
    goal_xtc = os.path.join(prot_dir, 'goal.xtc')  # small version, used for rmsd
    goal_prot_only = os.path.join(prot_dir, 'goal_prot.gro')  # needed for knn_rms
    init_prot_only = os.path.join(prot_dir, 'init_prot.gro')  # needed for contacts

    goal_bb_only = os.path.join(prot_dir, 'goal_bb.gro')  # needed for knn_rms
    # goal_bb_gro = os.path.join(prot_dir, 'goal_bb.gro')
    goal_bb_xtc = os.path.join(prot_dir, 'goal_bb.xtc')
    goal_angle_file = os.path.join(prot_dir, 'goal_angle.dat')
    goal_sincos_file = os.path.join(prot_dir, 'goal_sincos.dat')

    # I create two structures to reduce number input params in compute_metric
    # the more metrics we have in the future - the more parameters we have to track and pass
    goal_conf_files = {"goal_box_gro": goal,
                       "goal_prot_only_gro": goal_prot_only,
                       "goal_bb_only_gro": goal_bb_only,
                       "goal_prot_only_xtc": goal_xtc,
                       "goal_bb_xtc": goal_bb_xtc,
                       "angl_file_angl": goal_angle_file,
                       "sin_cos_file": goal_sincos_file,
                       "goal_top": topol_file_goal,
                       "goal_bb_ndx": goal_bb_ndx,
                       "goal_prot_ndx": ndx_file_goal}

    init_conf_files = {"init_top": topol_file_init,
                       "init_bb_ndx": init_bb_ndx,
                       "init_prot_ndx": ndx_file_init}

    # create prot_only init and goal
    gmx_trjconv(f=init, o=init_xtc, n=ndx_file_init)
    gmx_trjconv(f=goal, o=goal_xtc, n=ndx_file_goal)
    gmx_trjconv(f=goal, o=goal_prot_only, n=ndx_file_goal, s=goal)
    gmx_trjconv(f=goal_prot_only, o=goal_bb_only, n=goal_bb_ndx, s=goal_prot_only)
    gmx_trjconv(f=init, o=init_prot_only, n=ndx_file_init, s=init)
    gmx_trjconv(f=init_prot_only, o=init_xtc_bb, n=init_bb_ndx, s=init)
    gmx_trjconv(f=goal_prot_only, o=goal_bb_xtc, n=goal_bb_ndx, s=goal_prot_only)

    get_bb_to_angle_mdsctk(x=goal_bb_xtc, o=goal_angle_file)
    get_angle_to_sincos_mdsctk(i=goal_angle_file, o=goal_sincos_file)

    atom_num = get_atom_num(ndx_file_init)
    atom_num_bb = get_atom_num(goal_bb_ndx)
    angl_num = 2 * int(atom_num_bb / 3) - 2  # each bb amino acid has 3 atoms, thus 3 angles, we skip 1 since it is almost always 0.
    # In order to make plain you need three points, this is why you loose 2 elements. Last two do not have extra atoms to form a plain.

    with open(goal_sincos_file, 'rb') as file:
        initial_1d_array = np.frombuffer(file.read(), dtype=np.float64, count=-1)
    goal_angles = np.reshape(initial_1d_array, (-1, angl_num*2))[0]
    del file, initial_1d_array

    cont_dist = 3.0
    goal_ind = get_contat_profile_mdsctk(goal_prot_only, goal_xtc, ndx_file_goal, cont_dist)[1:]  # first is total num of contacts
    goal_contacts = np.zeros(atom_num * atom_num, dtype=np.bool)
    goal_contacts[goal_ind] = True
    del goal_ind

    h_pos_goal = parse_top_for_h(topol_file_goal)
    h_filter_goal = np.zeros(atom_num * atom_num, dtype=np.bool)
    for pos in h_pos_goal:
        h_filter_goal[(pos - 1) * atom_num:pos * atom_num] = True
    del pos
    goal_cont_h = np.logical_and(goal_contacts, h_filter_goal)

    h_pos_init = parse_top_for_h(topol_file_init)
    h_filter_init = np.zeros(atom_num * atom_num, dtype=np.bool)
    for pos in h_pos_init:
        h_filter_init[(pos - 1) * atom_num:pos * atom_num] = True
    del pos

    # usually h_filter_init is the same as h_filter_goal since they share same force field
    if np.sum(np.logical_xor(h_filter_init, h_filter_goal)) > 0:
        print('Warning, H positions in init and goal are different')
    del h_pos_goal, h_pos_init

    cpu_pool = mp.Pool(mp.cpu_count())

    goal_contacts_and_sum = np.sum(goal_contacts)
    goal_contacts_xor_sum = get_native_contacts(goal_prot_only, [goal_xtc], ndx_file_goal, goal_contacts,
                                                atom_num, cont_dist, np.logical_xor, pool=cpu_pool)[0]
    if goal_contacts_xor_sum != 0:
        raise Exception('goal.gro XOR goal.xtc is not 0 - they are different')
    else:
        del goal_contacts_xor_sum
    goal_contacts_and_h_sum = get_native_contacts(goal_prot_only, [goal_xtc], ndx_file_goal, goal_cont_h,
                                                  atom_num, cont_dist, np.logical_and, pool=cpu_pool)[0]
    # nat_contacts = np.sum(logic_fun(goal_contacts, init_contacts))

    if not os.path.exists(init_xtc) or not os.path.exists(goal_xtc) or \
            not os.path.exists(topol_file_init) or not os.path.exists(ndx_file_init):
        print('Copy initial and final state in to prot_dir')
        exit("Copy initial and final state in to prot_dir")

    work_dir = os.path.join(os.getcwd(), 'work_dir')  # either /dev/shm or os.getcwd()

    # counter = 0
    # work_dir = os.path.join('/dev/shm', 'work_dir_{}'.format(counter))  # either /dev/shm or os.getcwd()
    # while os.path.exists(work_dir):
    #     counter += 1
    #     work_dir = os.path.join('/dev/shm', 'work_dir_{}'.format(counter))  # either /dev/shm or os.getcwd()
    # del counter

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    print('Work dir: ', work_dir)

    if not os.path.exists(past_dir):
        os.makedirs(past_dir)

    print('Past dir: ', past_dir)

    simulation_temp = 350

    print('Information about the protein:\nIt contains {} atoms and {} hydrogen contacts'
          '\n{} phipsi angles is going to be used as for angle distance'
          '\nthere are {} protein-protein contacts with distance {}A\nand {} protein-protein-h contacts with distance {}A.'
          '\nSimulation temp is set to {}K'
          ''.format(atom_num, np.sum(goal_cont_h), angl_num, goal_contacts_and_sum, cont_dist,
                    goal_contacts_and_h_sum, cont_dist, simulation_temp))

    seed_start = 0
    seed_list = list(range(seed_start, tot_seeds+seed_start))
    del seed_start
    seed_dirs = get_seed_dirs(work_dir, seed_list, simulation_temp)
    # rm_seed_dirs(seed_dirs)

    if os.path.exists(os.path.join(os.getcwd(), 'local.comp')):
        use_mpi = False
    else:
        use_mpi = True

    scheduler = False
    if scheduler:
        use_mpi = True
        core_map = 16
        nomp = 2
        hostnames = False
    else:
        nomp = False
        if use_mpi:
            hostnames, core_map = parse_hostnames(tot_seeds)
        else:
            cpu_map = create_core_mapping(nseeds=tot_seeds)
            hostnames = False

    metric_names =      ['BBRMSD', 'AARMSD', 'ANGL', 'AND_H', 'AND', 'XOR']
    metric_allowed_sc = {'BBRMSD': 15, 'AARMSD': 20, 'ANGL': 10, 'AND_H': 5, 'AND': 5, 'XOR': 10}
    metrics_sequence =  ['AARMSD', 'BBRMSD']

    metric_rules = define_rules()

    cur_metric = 0
    cur_metric_name = metrics_sequence[cur_metric]
    guiding_metric = 0  # main metric to tack global progress

    num_metrics = len(metric_names)

    an_file = 'ambient.noise'
    err_mult = 0.8
    tol_error = check_precomputed_noize(an_file)
    noize_file = None
    if tol_error is None:
        goal_nz = os.path.join(prot_dir, 'folded_for_noise.gro')
        if hostnames:
            noize_file = gen_file_for_amb_noize(work_dir, seed_list, seed_dirs, ndx_file_goal,
                                                topol_file_goal, goal_nz, hostnames, core_map)
        else:
            # noize_file = gen_file_for_amb_noize(work_dir, goal_nz, seed_list, seed_dirs, ndx_file_goal, topol_file_goal, goal_nz)
            noize_file = gen_file_for_amb_noize(work_dir, seed_list, seed_dirs, ndx_file_goal, topol_file_goal, goal_nz)
        # 0 - rmsd, 1 - angles, 2 - h_contacts, 3 - full_contacts_xor, 4 - full_contacts_and
    if tol_error is None or len(tol_error) < num_metrics:
        if noize_file is None:
            noize_file = 'noise.xtc'
        goal_nz = os.path.join(prot_dir, 'folded_for_noise.gro')
        goal_prot_only_nz = os.path.join(prot_dir, 'goal_prot_nz.gro')
        goal_prot_only_nz_bb = os.path.join(prot_dir, 'goal_prot_nz_bb.xtc')
        noize_file_bb = os.path.join(prot_dir, 'goal_bb_nz.xtc')
        gmx_trjconv(f=goal_nz, o=goal_prot_only_nz, n=ndx_file_goal, s=goal_nz)
        gmx_trjconv(f=goal_prot_only_nz, o=goal_prot_only_nz_bb, n=goal_bb_ndx, s=goal_nz)
        goal_angle_file_nz = os.path.join(prot_dir, 'goal_angle_nz.dat')
        goal_sincos_file_nz = os.path.join(prot_dir, 'goal_sincos_nz.dat')
        goal_bb_xtc_nz = os.path.join(prot_dir, 'goal_bb_nz.xtc')
        gmx_trjconv(f=goal_nz, o=goal_bb_xtc_nz, n=goal_bb_ndx, s=goal_nz)
        gmx_trjconv(f=noize_file, o=noize_file_bb, n=goal_bb_ndx, s=goal_nz)
        goal_xtc_nz = os.path.join(prot_dir, 'goal_nz.xtc')
        gmx_trjconv(f=goal_nz, o=goal_xtc_nz, n=ndx_file_goal)
        get_bb_to_angle_mdsctk(x=goal_bb_xtc_nz, o=goal_angle_file_nz)
        get_angle_to_sincos_mdsctk(i=goal_angle_file_nz, o=goal_sincos_file_nz)
        with open(goal_sincos_file_nz, 'rb') as file:
            initial_1d_array = np.frombuffer(file.read(), dtype=np.float64, count=-1)
        goal_angles_nz = np.reshape(initial_1d_array, (-1, angl_num * 2))[0]
        del file, initial_1d_array
        goal_ind_nz = get_contat_profile_mdsctk(goal_prot_only, goal_xtc, ndx_file_goal, cont_dist)[1:]  # first is total num of contacts
        goal_contacts_nz = np.zeros(atom_num * atom_num, dtype=np.bool)
        goal_contacts_nz[goal_ind_nz] = True
        del goal_ind_nz

        h_pos_goal_nz = parse_top_for_h(topol_file_goal)
        h_filter_goal_nz = np.zeros(atom_num * atom_num, dtype=np.bool)
        for pos in h_pos_goal_nz:
            h_filter_goal_nz[(pos - 1) * atom_num:pos * atom_num] = True
        del h_pos_goal_nz, pos
        goal_cont_h_nz = np.logical_and(goal_contacts_nz, h_filter_goal_nz)

        goal_contacts_and_h_sum_nz = get_native_contacts(goal_prot_only_nz, [goal_xtc_nz], ndx_file_goal, goal_cont_h_nz,
                                                         atom_num, cont_dist, np.logical_and, pool=cpu_pool)[0]
        goal_contacts_and_sum_nz = np.sum(goal_contacts_nz)
        err_node_info = compute_init_metric(past_dir, tot_seeds, noize_file, noize_file_bb, angl_num,
                                            goal_angles_nz, goal_prot_only_nz, ndx_file_goal, goal_cont_h_nz, atom_num, cont_dist,
                                            h_filter_goal_nz, goal_contacts_nz, goal_contacts_and_h_sum_nz, goal_contacts_and_sum_nz,
                                            goal_conf_files)
        tol_error = dict()
        for metr_name in metric_names:
            tol_error[metr_name] = min([node['{}_to_goal'.format(metr_name)] for node in err_node_info]) * err_mult
        save_an_file(an_file, tol_error, metric_names)
        del err_node_info, metr_name
    del an_file, noize_file

    print('Done measuring ambient noise for folded state at {}K.\n'
          'Min result for {} seeds was multiplied by {}.\n'
          'BBRMSD noise was {:0.5f}A\n'
          'AARMSD noise was {:0.5f}A\n'
          'PhiPsi angle noise was {:0.5f}\n'
          'Contact distance noise with AND logical function for H contacts was {:.3f}\n'
          'Contact distance noise with AND logical function was {:.3f}\n'
          'Contact distance noise with XOR logical function was {:.3f}\n'
          ''.format(simulation_temp, tot_seeds, err_mult, tol_error['BBRMSD'], tol_error['AARMSD'], tol_error['ANGL'], tol_error['AND_H'],
                    tol_error['AND'], tol_error['XOR']))
    del err_mult
    node_info = compute_init_metric(past_dir, 1, init_xtc, init_xtc_bb, angl_num, goal_angles, init_prot_only,
                                    ndx_file_init, goal_cont_h, atom_num, cont_dist, h_filter_init, goal_contacts,
                                    goal_contacts_and_h_sum, goal_contacts_and_sum, goal_conf_files)

    print('Done measuring distance from initial state at {}K.\n'
          'BBRMSD dist: {:0.5f}A\n'
          'AARMSD dist: {:0.5f}A\n'
          'PhiPsi angle difference: {:0.5f}\n'
          'H contact disagreement (AND_H): {} of {}\n'
          'All contact disagreement (AND): {} of {}\n'
          'All contact disagreement (XOR): {}\n'.format(simulation_temp,
                                                        node_info['BBRMSD_to_goal'],
                                                        node_info['AARMSD_to_goal'],
                                                        node_info['ANGL_to_goal'],
                                                        node_info['AND_H_to_goal'], goal_contacts_and_h_sum,
                                                        node_info['AND_to_goal'], goal_contacts_and_sum,
                                                        node_info['XOR_to_goal']))

    print('Unfolded to noise ratio:\n'
          'BBRMSD : {:.5f}\n'
          'AARMSD : {:.5f}\n'
          'PhiPsi angles: {:.5f}\n'
          'H contact (AND_H) disagreement: {:.5f}\n'
          'All contact (AND) disagreement: {:.5f}\n'
          'All contact disagreement (XOR): {:.5f}\n'.format(node_info['BBRMSD_to_goal'] / tol_error['BBRMSD'] if tol_error['BBRMSD'] != 0 else float('inf'),
                                                            node_info['AARMSD_to_goal'] / tol_error['AARMSD'] if tol_error['AARMSD'] != 0 else float('inf'),
                                                            node_info['ANGL_to_goal'] / tol_error['ANGL'] if tol_error['ANGL'] != 0 else float('inf'),
                                                            node_info['AND_H_to_goal']/tol_error['AND_H'] if tol_error['AND_H'] != 0 else float('inf'),
                                                            node_info['AND_to_goal'] / tol_error['AND'] if tol_error['AND'] != 0 else float('inf'),
                                                            node_info['XOR_to_goal'] / tol_error['XOR'] if tol_error['XOR'] != 0 else float('inf')))

    # part of code used to study relation between contact distance and noise
    # f.write(
    #     '{} \n'.format(' '.join(str(elem) for elem in [cont_dist, node_info['AND_H_to_goal'], goal_contacts_and_h_sum,
    #     node_info['AND_H_to_goal'] / goal_contacts_and_h_sum, node_info['AND_to_goal'],
    #                                                    goal_contacts_and_sum,
    #                                                    node_info['AND_to_goal'] / goal_contacts_and_sum, node_info['XOR_to_goal'],
    #                                                    node_info['AND_H_to_goal'] / tol_error['AND_H'],
    #                                                    node_info['AND_to_goal'] / tol_error['AND'],
    #                                                    node_info['XOR_to_goal'] / tol_error['XOR']])))
    # print('done writing the file')
    # exit(22)
    # name_2_digest_map = dict()
    # digest_2_name_map = dict()
    # name_2_digest_map['s'] = get_digest('s')
    cur_hash_name = get_digest('s')
    # digest_2_name_map[name_2_digest_map['s']] = 's'

    main_dict = dict()
    main_dict[cur_hash_name] = node_info

    open_queue = list()
    heapq.heappush(open_queue, (node_info['{}_to_goal'.format(metric_names[0])], 0, cur_hash_name))  # metric_val, attempts, name
    ['BBRMSD', 'AARMSD', 'ANGL', 'AND_H', 'AND', 'XOR']
    init_metr = {'BBRMSD': node_info['BBRMSD_to_goal'], 'AARMSD': node_info['AARMSD_to_goal'], 'ANGL': node_info['ANGL_to_goal'],
                 'AND_H': node_info['AND_H_to_goal'], 'AND': node_info['AND_to_goal'], 'XOR': node_info['XOR_to_goal']}

    cp2(init_xtc[:-4] + '.gro', os.path.join(past_dir, cur_hash_name + '.gro'))
    cp2(init_xtc[:-4] + '.xtc', os.path.join(past_dir, cur_hash_name + '.xtc'))
    # copy_queue.put_nowait((init_xtc[:-4] + '.gro', os.path.join(past_dir, name_2_digest_map['s'] + '.gro')))
    # copy_queue.put_nowait((init_xtc[:-4] + '.xtc', os.path.join(past_dir, name_2_digest_map['s'] + '.xtc')))
    # copy_queue.put_nowait(None)

    visited_queue = list()
    skipped_counter = 0

    combined_pg = os.path.join(work_dir, "out.xtc")
    combined_pg_bb = os.path.join(work_dir, "out_bb.xtc")
    temp_xtc_file = os.path.join(work_dir, "temp.xtc")
    temp_xtc_file_bb = os.path.join(work_dir, "temp_bb.xtc")

    loop_start = time.perf_counter()

    # info_form_str = 'n:{}\db_input_thread:{:.4f}\tg:{:.4f}\ts:{}\tq:{}\tv:{}\tl:{:.2f}s\tc:{:.2f}s'
    info_form_str = 'o_q:{:<5}  v_q:{:<3}  s:{:<3}  grm:{:6.2f}  gan:{:6.2f}  gah:{:<4}  gad:{:<4}  gxo:{:<4}  ' \
                    't:{:5.2f}s  gbrb:{:.3f}  gbr:{:.3f}  gba:{:.3f}  gc:{:<2}  ns:{:3.1f}  sc:{}'
    #  node_info['rmds_total'], node_info['rmds_to_goal'], skipped_counter, len(open_queue), len(visited_queue),
    # loop_end - loop_start, best_so_far, global_best_so_far, greed_count, greed_mult, seed_change_counter,
    # node_info['nat_cont_to_goal']))
    # info_form_str.format(len(open_queue), len(visited_queue), skipped_counter, node_info['RMSD_to_goal'],
    # node_info['ANGL_to_goal'], node_info['AND_H_to_goal'],
    #                      node_info['AND_to_goal']), node_info['XOR_to_goal'], loop_end - loop_start, best_so_far[1],
    #                      best_so_far[0], greed_count, greed_mult, seed_change_counter)
    under_form_str = '{}_{}'

    greed_mult = 1.0
    greed_count = 0

    # con, dbname = get_db_con(tot_seeds)
    # insert_into_main_stor(con, node_info, greed_count, name_2_digest_map['s'], 's')
    db_input_queue.put_nowait((insert_into_main_stor, (node_info, greed_count, cur_hash_name, 's')))

    node_max_att = 4

    seed_change_counter = 0
    # change_metrics_limit = 3  # how many seed changes(20 iter per change) with no problems we have to have to change cur metricss

    # search LMA in the code
    # seed_change_limit = 1000
    # local_minimum_counter = 0
    # local_minim_names = list()

    # nmr_structure_switch = 2  # 0 for nmr, 1 for relaxed, 2 for heated

    best_so_far = {metr: node_info['{}_to_goal'.format(metr)] for metr in metric_names}
    print(best_so_far)
    best_so_far_name = {metr: cur_hash_name for metr in metric_names}
    # global_best_so_far = best_so_far

    Path(combined_pg).touch()
    Path(combined_pg_bb).touch()
    Path(temp_xtc_file).touch()
    Path(temp_xtc_file_bb).touch()
    if os.path.exists('./local_min.xtc'):
        os.remove(('./local_min.xtc'))

    compute_all_at_once = True
    counter_since_seed_changed = 0

    recover = False  # STOP! before changing this toggle read bellow:
    # 1. Make backup of your pickles
    # 2. Remember number of the last good db - this name should always be the last one
    # There was no proper testing of this functionality and backups may overwrite last good state
    # Backups rely on time and number of steps, but if you have too fast/slow I/O - everything may go wrong. Thus do the pickle backup.
    if recover:  # this can (and should) be done in parallel or instead of most var initialization (much earlier)
        visited_queue, open_queue, main_dict = main_state_recover()
        prev_state = supp_state_recover()
        tol_error, seed_list, seed_dirs, seed_change_counter, skipped_counter, \
        cur_metric_name, cur_metric, counter_since_seed_changed, guiding_metric, greed_mult, \
        best_so_far_name, best_so_far, greed_count, rules = prev_state
        del prev_state
        copy_old_db(list(main_dict.keys()), visited_queue[-3:].copy()[::-1], open_queue[0][2], greed_count-1)

    # try:
    # aa = 0
    iter_from_bak = 0
    time_for_backup = False
    bak_time_check = time.perf_counter()
    while len(open_queue) > 0:  # and aa < 137:
        gc.collect()
        # if not aa % 10:
        #     # Prints out a summary of the large objects
        #     summary.print_(summary.summarize(muppy.get_objects()))
        # aa +=1
        new_elem = heapq.heappop(open_queue)  # tot_dist, att, name
        tot_dist, att, cur_hash_name = new_elem
        del new_elem
        if counter_since_seed_changed:  # you may disable this check, it was here to track nodes with the same name.
            if check_dupl(cur_hash_name, visited_queue[-counter_since_seed_changed:]):
                continue
        # however, if you see nodes with the same name - check real name and if it is different - change hashing function
        # much
        counter_since_seed_changed += 1

        node_info = main_dict[cur_hash_name]
        cur_name = zlib.decompress(node_info['native_name']).decode()
        # cur_file = os.path.join(past_dir, node_info['digest_name'])

        visited_queue.append((tot_dist, att+1, cur_hash_name))  # TODO: trim it when size > 500 by 300, update tot_trim
        del tot_dist, att

        db_input_queue.put_nowait((insert_into_visited, (cur_hash_name, greed_count)))
        db_input_queue.put_nowait((insert_into_log, ('result', cur_hash_name, 'WQ', 'VIZ', best_so_far, greed_count, greed_mult,
                                                     node_info['{}_dist_total'.format(cur_metric_name)],
                                                     node_info['{}_to_goal'.format(cur_metric_name)], cur_metric_name)))
        # insert_into_visited(con, cur_name, greed_count)
        # insert_into_log(con, 'result', cur_name, 'WQ', 'VIZ', best_so_far, greed_count, greed_mult, node_info['{}_dist_total'.
        #                  format(cur_metric_name)], node_info['{}_to_goal'.format(cur_metric_name)])
        loop_end = time.perf_counter()

        # print_queue.put_nowait((info_form_str,
        #                         ((len(open_queue), len(visited_queue), skipped_counter, node_info['AARMSD_to_goal'],
        #                           node_info['ANGL_to_goal'], node_info['AND_H_to_goal'], node_info['AND_to_goal'],
        #                           node_info['XOR_to_goal'], loop_end - loop_start, best_so_far["BBRMSD"], best_so_far["AARMSD"],
        #                           best_so_far["ANGL"], greed_count, greed_mult, seed_change_counter))))
        print(info_form_str.format(len(open_queue), len(visited_queue), skipped_counter, node_info['AARMSD_to_goal'],
                                  node_info['ANGL_to_goal'], node_info['AND_H_to_goal'], node_info['AND_to_goal'],
                                  node_info['XOR_to_goal'], loop_end - loop_start, best_so_far["BBRMSD"], best_so_far["AARMSD"],
                                  best_so_far["ANGL"], greed_count, greed_mult, seed_change_counter))

        # if node_info['ANGL_to_goal'] < best_so_far[1]:
        #     print('BSF:')
        #     print(best_so_far)
        #     print('Cur node info ANGL'.format(node_info['ANGL_to_goal']))
        #     print('Cur node info name'.format(cur_name))
        #     raise Exception('Error in best so far')

        loop_start = time.perf_counter()
        if not use_mpi:
            pid_arr, files_for_trjcat, recent_filenames, recent_n2d, recent_d2n = compute_on_local_machine(cpu_map, seed_list, cur_name,
                                                                                                           past_dir, work_dir, seed_dirs,
                                                                                                           topol_file_init, ndx_file_init,
                                                                                                           cur_hash_name)
        else:
            pid_arr, files_for_trjcat, recent_filenames, recent_n2d, recent_d2n = compute_with_mpi(seed_list, cur_name, past_dir, work_dir,
                                                                                                   seed_dirs, topol_file_init,
                                                                                                   ndx_file_init,
                                                                                                   cur_hash_name, tot_seeds, hostnames,
                                                                                                   core_map, scheduler, nomp)

        # update map
        # name_2_digest_map.update(recent_n2d)
        # digest_2_name_map.update(recent_d2n)
        del recent_filenames, recent_n2d, recent_d2n

        os.remove(combined_pg)
        os.remove(combined_pg_bb)
        gmx_trjcat(f=['{}.xtc'.format(os.path.join(past_dir, cur_hash_name)), goal_xtc],
                   o=combined_pg, n=ndx_file_init, cat=True, vel=False, sort=False, overwrite=True)

        gmx_trjcat(f=['{}.xtc'.format(os.path.join(past_dir, cur_hash_name)), goal_xtc],
                   o=combined_pg_bb, n=init_bb_ndx, cat=True, vel=False, sort=False, overwrite=True)

        [proc.join() for proc in pid_arr]
        del pid_arr

        if compute_all_at_once or cur_metric < 2:
            os.remove(temp_xtc_file)
            gmx_trjcat(f=files_for_trjcat, o=temp_xtc_file, n=ndx_file_init, cat=True, vel=False, sort=False, overwrite=True)
            gmx_trjcat(f=temp_xtc_file, o=temp_xtc_file_bb, n=init_bb_ndx, cat=True, vel=False, sort=False, overwrite=True)

        new_nodes_names = [under_form_str.format(cur_name, seed_name) for seed_name in seed_list]
        # for i, node in enumerate(new_nodes):
        #     new_nodes[i]['digest_name'] = get_digest(new_nodes_names[i])
        #     # new_nodes[i]['native_name'] = new_nodes_names[i]
        #     new_nodes[i]['native_name'] = zlib.compress(new_nodes_names[i].encode(), 9)
        # del node, i
        new_nodes, metric_to_goal, metric_form_prev, metric_to_tot = compute_metric(past_dir, new_nodes_names, tot_seeds, combined_pg,
                                                                                    combined_pg_bb, temp_xtc_file, temp_xtc_file_bb,
                                                                                    node_info, angl_num, goal_angles, init_prot_only,
                                                                                    files_for_trjcat, ndx_file_init, goal_cont_h,
                                                                                    atom_num, cont_dist, h_filter_init, goal_contacts,
                                                                                    cur_metric, goal_contacts_and_h_sum,
                                                                                    goal_contacts_and_sum, goal_conf_files,
                                                                                    cpu_pool=cpu_pool,
                                                                                    compute_all_at_once=compute_all_at_once)
        del files_for_trjcat

        new_filtered = list()
        for i in range(tot_seeds):
            # if seed_change_counter:
            #     local_minim_names.append(seed_name)

            # MAIN INSERT  new_nodes, metric_form_prev, metric_to_goal, metric_to_tot
            # we have two conditions to get intro the queue:
            # 1st - get better than the best result (obvious)
            # 2nd - we have to make big enough step from the previous point
            # AND this step should bring us closer to the goal 1/2 of just a noise
            if (metric_form_prev[i] > tol_error[cur_metric_name]
                and metric_to_goal[i] - node_info['{}_to_goal'.format(cur_metric_name)] < tol_error[cur_metric_name] / 2) \
                    or metric_to_goal[i] <= best_so_far[cur_metric_name] or (len(open_queue) < 20 and len(visited_queue) < 1000):
                # LMA - this approach is currently frozen since it did not show any benefits with RMSD,
                # but was never adapted to multiple metrics
                # if check_local_minimum(temp_xtc_file, goal_prot_only, tol_error):
                # else:
                #     print('point was on path to local minimum')

                heapq.heappush(open_queue, (greed_mult * metric_to_tot[i] + metric_to_goal[i], 0, new_nodes[i]['digest_name']))
                new_filtered.append((greed_mult * metric_to_tot[i] + metric_to_goal[i], 0, new_nodes[i]['digest_name']))
                # insert_into_main_stor(con, new_nodes[i], greed_count,
                # name_2_digest_map[new_nodes_names[i]], new_nodes_names[i])
                db_input_queue.put_nowait((insert_into_main_stor,
                                          (new_nodes[i], greed_count, new_nodes[i]['digest_name'], new_nodes_names[i])))
                main_dict[new_nodes[i]['digest_name']] = new_nodes[i]
            else:
                skipped_counter += 1
                # insert_into_log(con, 'skip', cur_name, '', 'SKIP', best_so_far, greed_count,
                # greed_mult, metric_form_prev[i], metric_form_prev[i])
                db_input_queue.put_nowait((insert_into_log, ('skip', cur_hash_name, '', 'SKIP', best_so_far, greed_count,
                                                             greed_mult, metric_form_prev[i], metric_to_goal[i], cur_metric_name)))
        db_input_queue.put_nowait((insert_into_log, ('current', cur_hash_name, '', 'WQ', best_so_far, greed_count,
                                                     greed_mult, metric_form_prev, metric_to_goal, cur_metric_name)))
        del metric_to_tot, metric_form_prev, i, new_nodes_names

        if compute_all_at_once:
            for metr in metric_names:
                if metr != cur_metric_name:
                    min_val = min([node['{}_to_goal'.format(metr)] for node in new_nodes])
                    if best_so_far[metr] > min_val:
                        # print('bsf["{}"]={:.4f}, min={:.4f}'.
                        # format(metr, best_so_far[metric_names.index(metr)], min_val), end=' ')
                        best_so_far[metr] = min_val
                    del min_val
                # else:
                    # print('skipping "{}"'.format(metr), end=' ')
            del metr
        # print()
        if best_so_far[metric_names[guiding_metric]] > new_nodes[metric_to_goal.index(min(metric_to_goal))]['{}_to_goal'.format(metric_names[guiding_metric])]:
            seed_change_counter = 0

        if best_so_far[cur_metric_name] > min(metric_to_goal):
            best_so_far_new = min(metric_to_goal)
            best_so_far[cur_metric_name] = best_so_far_new
            best_so_far_name[cur_metric_name] = new_nodes[metric_to_goal.index(best_so_far_new)]['digest_name']
            db_input_queue.put_nowait((insert_into_log,
                                       ('prom_O', best_so_far_name[cur_metric_name], '', '', best_so_far, greed_count, greed_mult,
                                        new_nodes[metric_to_goal.index(best_so_far_new)]['{}_from_prev'.format(cur_metric_name)],
                                        new_nodes[metric_to_goal.index(best_so_far_new)]['{}_to_goal'.format(cur_metric_name)],
                                        cur_metric_name)))
            if guiding_metric == cur_metric or best_so_far[metric_names[guiding_metric]] >= new_nodes[metric_to_goal.index(best_so_far_new)]['{}_to_goal'.format(metric_names[guiding_metric])]:
                for i in range(num_metrics):
                    if i != cur_metric:
                        best_so_far_name[metric_names[i]] = best_so_far_name[cur_metric_name]
                        best_so_far[i] = new_nodes[metric_to_goal.index(best_so_far_new)]['{}_to_goal'.format(metric_names[i])]
                del i
            seed_change_counter = 0

            # local_minim_names = list()  # search for LMA
            # if global_best_so_far[cur_metric] > best_so_far_new:
            #     global_best_so_far[cur_metric] = best_so_far_new

            # This code is for multiple stage folding. Code has to be adapted for several metrics.
            # if len(visited_queue) > 1 and global_best_so_far < visited_queue[1][2]/5 and nmr_structure_switch == 1:
            #     print('Changing goal to nmr structure')
            #     cp2(os.path.join(prot_dir, 'nmr.gro'), goal)
            #     gmx_trjconv(f=goal, o=goal_xtc, n=ndx_file)
            #     gmx_trjconv(f=goal, o=goal_prot_only, n=ndx_file, s=goal)
            #     open_queue = recompute_rmsd_for_openq(open_queue, goal_xtc, name_2_digest_map, past_dir,
            #     goal_prot_only, greed_mult)
            #     best_so_far = open_queue[-1][2]
            #     nmr_structure_switch = 0
            # elif len(visited_queue) > 1 and global_best_so_far < visited_queue[1][2]/3 and nmr_structure_switch == 2:
            #     print('Changing goal to relaxed structure')
            #     cp2(os.path.join(prot_dir, 'relaxed.gro'), goal)
            #     gmx_trjconv(f=goal, o=goal_xtc, n=ndx_file)
            #     gmx_trjconv(f=goal, o=goal_prot_only, n=ndx_file, s=goal)
            #     open_queue = recompute_rmsd_for_openq(open_queue, goal_xtc, name_2_digest_map, past_dir,
            #     goal_prot_only, greed_mult)
            #     best_so_far = open_queue[-1][2]
            #     nmr_structure_switch = 1

            # This is part of local minimum approach (LMA) search for LMA in this code
            # if os.path.exists('./local_minim_bas.xtc'):
            #     os.remove('./local_minim_bas.xtc')
            del best_so_far_new
            if greed_mult < 1.0:  # perfect place to optimize queue rebuild
                greed_count = max(0, 10 * (greed_count // 10) - 8)
                if 100 < greed_count < 110:
                    greed_count = 101
                else:
                    greed_mult = min(1.001 - min(1.0, (greed_count // 10) / 10), 1.0)
                    open_queue = queue_rebuild(None, open_queue, main_dict, greed_mult, cur_metric_name, sep_proc=False)
            else:
                greed_count = 0
        else:
            greed_count += 1

            if greed_count in range(10, 101, 10):
                # open_queue = rebuild_queue.get(timeout=1800)[0]  # 30min
                open_queue = rebuild_queue.get()[0]  # 30min
                if new_filtered:
                    for elem in new_filtered:
                        heapq.heappush(open_queue, elem)
                # cur_metric = metric_names.index(cur_metric_name)
                del rebuild_queue
                # if not isinstance(rebuild_queue_process, mp.Process):
                #     a=8
                rebuild_queue_process.join()

            elif greed_count == 121:
                seeds_next = get_new_seeds(seed_list)
                seed_change_counter += 1
                seed_dirs_next = get_seed_dirs(work_dir, seeds_next, simulation_temp)
                # previously I passed here "seed_dirs", but decided to save RAM
                if seed_change_counter > metric_allowed_sc[cur_metric_name]:
                    new_metr_name = select_metrics_by_snr(new_nodes, node_info, metric_names, tol_error,
                                                          compute_all_at_once, metrics_sequence, cur_metric_name)
                    rebuild_queue = mp.Queue()
                    # open_queue = queue_rebuild(None, open_queue, main_dict, greed_mult, new_metr_name, sep_proc=False)
                    rebuild_queue_process = mp.Process(target=queue_rebuild,
                                                       args=(rebuild_queue, open_queue, main_dict, greed_mult, new_metr_name))
                    # if not isinstance(rebuild_queue_process, mp.Process):
                    #     a = 8
                    rebuild_queue_process.start()
                    del new_metr_name
                # TODO: local minimum has to be rethought and rewritten.
                #  At this point (before multiple metrics) experiments show that is does not give any benefits
                # if seed_change_counter == seed_change_limit:
                #     seed_change_counter = 0
                #     greed_count = 112
                #     open_queue = proc_local_minim(open_queue, best_so_far_name[cur_metric_name], tol_error, ndx_file_init,
                #     name_2_digest_map, goal_prot_only, local_minim_names)
                #     local_minim_names = list()
                #     best_so_far[cur_metric_name] = (init_distance[cur_metric] + best_so_far[cur_metric_name])/2
                #     local_minimum_counter += 1
                #     continue
        del metric_to_goal

        if greed_count in range(9, 100, 10):
            rebuild_queue = mp.Queue()
            greed_mult = min(1.001 - (greed_count+1) / 100, 1.0)
            rebuild_queue_process = mp.Process(target=queue_rebuild, args=(rebuild_queue, open_queue, main_dict,
                                                                           greed_mult, cur_metric_name))
            rebuild_queue_process.start()
        elif greed_count == 122:
            greed_count = 102
            if seed_change_counter > metric_allowed_sc[cur_metric_name]:
                print('Switching metric from {} to '.format(cur_metric_name), end='')
                open_queue, cur_metric_name = rebuild_queue.get()  # 30min
                # open_queue, cur_metric_name = rebuild_queue.get(timeout=1800)  # 30min
                print(cur_metric_name)
                cur_metric = metric_names.index(cur_metric_name)
                del rebuild_queue
                rebuild_queue_process.join()
                extra_elem_q = queue_rebuild(None, new_filtered, main_dict, greed_mult, cur_metric_name, sep_proc=False)
                for elem in extra_elem_q:
                    heapq.heappush(open_queue, elem)
                del extra_elem_q, elem
                seed_change_counter = 0
                # greed_count = 102

            if seeds_next:
                seed_list = seeds_next
                rm_seed_dirs(seed_dirs)
                seed_dirs = seed_dirs_next
                res_arr = second_chance(open_queue[0:min(len(open_queue)-1, max(40, 4*counter_since_seed_changed))],
                                        visited_queue[min(-1, -counter_since_seed_changed):],
                                        best_so_far_name, cur_metric, main_dict, node_max_att,
                                        cur_metric_name, best_so_far, tol_error, greed_mult)
                counter_since_seed_changed = 0
                for elem in res_arr:
                    heapq.heappush(open_queue, elem)
                    # print(elem)
                    db_input_queue.put_nowait((insert_into_log,
                                               ('result', cur_hash_name, 'VIZ', 'WQ', best_so_far, greed_count, greed_mult,
                                                main_dict[elem[2]]['{}_from_prev'.format(cur_metric_name)],
                                                main_dict[elem[2]]['{}_to_goal'.format(cur_metric_name)], cur_metric_name)))
            else:
                print('\nOUT OF SEEDS\n')
                greed_count = 102  # will be changed soon
            del seeds_next, seed_dirs_next
        del cur_hash_name, cur_name, new_nodes, node_info
        new_filtered.clear()

        metric_rules, metrics_sequence, switch_metric = check_rules(metrics_sequence, metric_rules, best_so_far, init_metr, metric_names, greed_count)
        if switch_metric is not None:
            print('Switching metric because of the rule')
            greed_mult = min(1.001 - (greed_count + 1) / 100, 1.0)
            open_queue = queue_rebuild(None, open_queue, main_dict, greed_mult, switch_metric, sep_proc=False)
            seed_change_counter = 0

        iter_from_bak += 1
        if loop_start - bak_time_check > 60*60 and not time_for_backup:  # every hour
            if iter_from_bak < 1000:  # expected value 240 - means that we are computing (on 32 cores), but not reading from ./past, typical read speed 10 000 iterations/hour (for non SSD)
                time_for_backup = True
            else:
                iter_from_bak = 0
                bak_time_check = loop_start

        if time_for_backup and (greed_count in range(104, 109) or greed_count in range(113, 117) or greed_count in range(93, 97)):
            try:
                main_state_backup((visited_queue, open_queue, main_dict))
                supp_state_backup((tol_error, seed_list, seed_dirs, seed_change_counter, skipped_counter, cur_metric_name,
                                   cur_metric, counter_since_seed_changed, guiding_metric, greed_mult,
                                 best_so_far_name, best_so_far, greed_count, metric_rules))
            except Exception as e:
                print('Error during the backup:')
                print(e)

            time_for_backup = False
            bak_time_check = time.perf_counter()
            iter_from_bak = 0


    # except (KeyboardInterrupt, Exception) as e:
    #     print('Got exception: ', e)
    #     exc_type, exc_obj, exc_tb = sys.exc_info()
    #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    #     print(exc_type, fname, exc_tb.tb_lineno)
    #     # print('Dumping work_queue')
    #     # dump_the_queue('work_queue.txt', open_queue, visited_queue, init_rmsd, tol_error, skipped_counter)
    #     # print('Dumping visited_queue')
    #     # dump_the_queue('visited_queue.txt', visited_queue, visited_queue, init_rmsd, tol_error, skipped_counter)
    #     # print('Done dumping ')
    #     # exit(-1)
    #
    #     # if keyboard.is_pressed('md_process'):
    #     #     print('Dumping ')
    #     #     dump_the_queue('work_queue.txt', open_queue, visited_queue, init_rmsd, tol_error, skipped_counter)
    #     #     print('Dumping ')
    #     #     dump_the_queue('visited_queue.txt', visited_queue, visited_queue, init_rmsd, tol_error, skipped_counter)
    #     #     print('Done dumping ')
    #
    #     # ne = open_queue[0]
    #     # trav = ne[1]
    #     # to_goal = ne[2]
    #     # sds = ne[3]
    #     # tot_points = len(sds.split("_")) - 1
    #     # from_prev_dist, prev_goal_dist = current_job[1], current_job[2]
    #     # trav_from_prev = trav - from_prev_dist
    #     # coef_1 = 1 - to_goal / init_rmsd
    #     # coef_1_a = coef_1 / tot_points if tot_points != 0 else 9999
    #     # deriv = (prev_goal_dist - to_goal) / trav_from_prev  # this cannot be zero
    #     # full_line = '{:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {}\n'.format(trav,
    #     #                                                                 to_goal,
    #     #                                                                 trav_from_prev,
    #     #                                                                 coef_1,
    #     #                                                                 coef_1_a,
    #     #                                                                 deriv,
    #     #                                                                 sds)
    #     # file.write(full_line)
    #
    #     # check_end = time.perf_counter()
    #
    # print('We are finally done with search.')
    # print('Current queue size: ', len(open_queue))
    # print('Current visited_queue queue: ', len(visited_queue))
    # # dump_the_queue('work_queue.txt', open_queue, visited_queue, init_rmsd, tol_error, skipped_counter)
    # # dump_the_queue('visited_queue.txt', visited_queue, visited_queue, init_rmsd, tol_error, skipped_counter)
