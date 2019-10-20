"""This file contains functions to compute various metric distances.

.. module:: GMDA_main
    :platform: linux

.. moduleauthor:: Ivan Syzonenko <is2k@mtmail.mtsu.edu>
"""
__license__ = "MIT"
__docformat__ = 'reStructuredText'


import numpy as np
import os
import subprocess
import multiprocessing as mp
from scipy.sparse import csc_matrix, save_npz, load_npz
import zlib
from typing import NoReturn
# from shutil import copy2 as cp2

from helper_funcs import get_digest
from gmx_wrappers import gmx_grompp, gmx_mdrun, gmx_trjcat, gmx_trjconv, gmx_mdrun_mpi
# from gen_mdp import get_mdp


def get_knn_dist_mdsctk(ref_file: str, fitfile: str, topology: str) -> list:
    """'knn_rms' - MDSCTK tool - computes RMSD between two (or more) structures

    Args:
        :param str ref_file: reference file - .xtc or .gro filename
        :param str fitfile: .xtc or .gro filename - structure will be centered according to the fitfile and used in distance computation
        :param str topology: .top topology file of the simulation box

    Returns:
        :return: list of RMSD distances from all frames to the goal
        :rtype: list
    """
    if os.path.exists(os.path.join(os.getcwd(), 'local.comp')):
        mdsctk_bash = 'source /opt/mdsctk/MDSCTK.bash ; '  # need this since load_envbash does not work
    else:
        mdsctk_bash = 'source ./mdsctk/MDSCTK.bash ; '  # need this since load_envbash does not work

    command = '{} knn_rms -s {} -p {} -r {} -f {}'.format(mdsctk_bash, 0, topology, ref_file, fitfile)
    proc_obj = subprocess.Popen(os.path.expandvars(command), stdout=subprocess.PIPE, shell=True, stderr=None)
    try:
        output, error = proc_obj.communicate()
    except Exception as e:
        print(e)
        return None
    if error:
        error = error.decode("utf-8")
        if 'error' in error.lower():
            print(error)
    if output:
        output = output.decode("utf-8")
        if 'error' in output.lower():
            print(output)
    dist_arr = np.fromfile('distances.dat', dtype=np.double)
    os.remove('distances.dat')
    os.remove('indices.dat')

    return dist_arr.tolist()


def get_contat_profile_mdsctk(ref_file: str, fitfile: str, index: str, dist: float = 2.7) -> np.ndarray:
    """'contact_profile' - MDSCTK tool - computes number of contacts between two (or more) structures

    Args:
        :param str ref_file: reference file - .xtc or .gro filename
        :param str fitfile: .xtc or .gro filename - structure will be centered according
         to the fitfile and used in distance computation
        :param str index: .ndx file to compute distance among particular atoms
        :param floatdist: in Angstroms - how close should two atoms be, so treat them as a contact

    Returns:
        :return: ndarray, first value - number of indices with contacts, next N indices are atoms with contact
        :rtype np.ndarray
    """
    if os.path.exists(os.path.join(os.getcwd(), 'local.comp')):
        mdsctk_bash = 'source /opt/mdsctk/MDSCTK.bash ; '  # need this since load_envbash does not work
    else:
        mdsctk_bash = 'source ./mdsctk/MDSCTK.bash ; '  # need this since load_envbash does not work

    slash_pos = fitfile.rfind('/')
    if slash_pos >= 0:
        unique_name = '{}/{}.svi'.format(fitfile[:slash_pos], fitfile.split('/')[-1].split('.')[0])
    else:
        unique_name = '{}.svi'.format(fitfile.split('/')[-1].split('.')[0])
    command = '{} contact_profile -p {} -x {} -n {} -e {} -i {} -d /dev/null 2>/dev/null 1>/dev/null'.format(
        mdsctk_bash, ref_file, fitfile, index, dist, unique_name)
    proc_obj = subprocess.Popen(os.path.expandvars(command), stdout=None, shell=True, stderr=None)
    try:
        output, error = proc_obj.communicate()
    except Exception as e:
        print(command)
        print(e)
        return None
    if error:
        error = error.decode("utf-8")
        if 'error' in error.lower():
            print(command)
            print(error)
    if output:
        output = output.decode("utf-8")
        if 'error' in output.lower():
            print(command)
            print(output)
    cont_arr = np.fromfile(unique_name, dtype=np.uint32)

    os.remove(unique_name)

    return cont_arr


def get_bb_to_angle_mdsctk(x: str = 'noise_bb.xtc', o: str = 'noise_angle.dat') -> NoReturn:
    """'bb_xtc_to_phipsi' - MDSCTK tool - takes backbone structure and computes dihedral angles between atoms

    Args:
        :param str x: backbone input trajectory
        :param str o: filename of the binary C array

    Returns:
    Generates a file with dihedral angles.
    """
    if os.path.exists(os.path.join(os.getcwd(), 'local.comp')):
        mdsctk_bash = 'source /opt/mdsctk/MDSCTK.bash ; '  # need this since load_envbash does not work
    else:
        mdsctk_bash = 'source ./mdsctk/MDSCTK.bash ; '  # need this since load_envbash does not work
    #  bb_xtc_to_phipsi -x traj_bb_315.xtc -o angles_bb_315.dat
    command = '{} bb_xtc_to_phipsi -x {} -o {} 2>/dev/null 1>/dev/null'.format(
        mdsctk_bash, x, o)
    proc_obj = subprocess.Popen(
        os.path.expandvars(command), stdout=None, shell=True, stderr=None)
    # proc_obj = subprocess.Popen(os.path.expandvars(command), stdout=subprocess.PIPE, shell=True, stderr=None)
    try:
        output, error = proc_obj.communicate()
    except Exception as e:
        print(command)
        # print(e)
        raise Exception(e)
    if error:
        error = error.decode("utf-8")
        if 'error' in error.lower():
            print(command)
            print(error)
    if output:
        output = output.decode("utf-8")
        if 'error' in output.lower():
            print(command)
            print(output)


def get_angle_to_sincos_mdsctk(i: str='noise_angle.dat', o: str='noise_sincos.dat') -> NoReturn:
    """'angles_to_sincos' - MDSCTK tool - converts dihedrals into sin/cos values

    Args:
        :param str i: filename that contains angle values in the  binary form
        :param str o: filename that contains sin/cos values in the  binary form

    Returns:
    Generates file with sin/cos values.
    """
    if os.path.exists(os.path.join(os.getcwd(), 'local.comp')):
        mdsctk_bash = 'source /opt/mdsctk/MDSCTK.bash ; '  # need this since load_envbash does not work
    else:
        mdsctk_bash = 'source ./mdsctk/MDSCTK.bash ; '  # need this since load_envbash does not work
    # angles_to_sincos -i angles_bb_315.dat -o sincos_bb_315.dat
    command = '{} angles_to_sincos -i {} -o {} 2>/dev/null 1>/dev/null'.format(
        mdsctk_bash, i, o)
    proc_obj = subprocess.Popen(
        os.path.expandvars(command), stdout=None, shell=True, stderr=None)
    # proc_obj = subprocess.Popen(os.path.expandvars(command), stdout=subprocess.PIPE, shell=True, stderr=None)
    try:
        output, error = proc_obj.communicate()
    except Exception as e:
        print(command)
        # print(e)
        raise Exception(e)
    if error:
        error = error.decode("utf-8")
        if 'error' in error.lower():
            print(command)
            print(error)
    if output:
        output = output.decode("utf-8")
        if 'error' in output.lower():
            print(command)
            print(output)


def gen_file_for_amb_noize(work_dir: str, seeds: int, seed_dirs: dict, ndx_file: str, top_file: str,
                           goal_file: str = 'folded_for_noise.gro', hostnames: list = None, cpu_map: list = None) -> str:
    """Performs simulation of the NMR (not unfolded) conformation to measure ambient vibrations

    Args:
        :param str work_dir: path to the working directory
        :param int seeds: number of seed in the current run
        :param dict seed_dirs: paths to directories where emulation is performed with particular seed
        :param str ndx_file: index file to extract only specific atoms (strip water)
        :param str top_file: .top topology file of the simulation box
        :param str goal_file: goal (typically NMR) conformation
        :param list hostnames: for MPI, to perform parallel computation
        :param listcpu_map: number of cores for particular task (seed)

    Returns:
        :return: filename which contains all seed simulations concatenated
        :rtype: str

    Generates a file with trajectories from the goal.
    """
    # if file ambient.rmsd found, read it

    temp_xtc_file = 'noise.xtc'
    #  generate and save if not found
    if temp_xtc_file not in os.walk(".").__next__()[2]:
        pid_arr = list()
        for i, seed in enumerate(seeds):

            gmx_grompp(work_dir, seed, top_file,
                       goal_file[:-4])  # TODO: update filenames

            if hostnames:
                md_process = mp.Process(target=gmx_mdrun_mpi,
                                        args=(work_dir, seed, os.path.join(seed_dirs[seed], 'md.gro'), hostnames[i], cpu_map[i]))
                # gmx_mdrun_mpi(work_dir, seed, seed_dirs[seed] + '/md.gro', hostnames[i], cpu_map[i])
            else:
                md_process = mp.Process(target=gmx_mdrun, args=(work_dir, seed, os.path.join(seed_dirs[seed], 'md.gro')))
                # gmx_mdrun(work_dir, seed, seed_dirs[seed] + '/md.gro')
            md_process.start()
            pid_arr.append(md_process)
        [proc.join() for proc in pid_arr]
        for i, seed in enumerate(seeds):
            gmx_trjconv(
                f=os.path.join(seed_dirs[seed], 'md.xtc'),
                o=os.path.join(seed_dirs[seed], 'md_prot.xtc'),
                n=ndx_file,
                b=1)  # , dump=20

        results_arr = list(os.path.join(os.path.join(work_dir, str(seed)), 'md_prot.xtc') for seed in seeds)
        gmx_trjcat(f=results_arr, o=temp_xtc_file, n=ndx_file, cat=True, vel=False, sort=False, overwrite=True)

    return temp_xtc_file


# def get_ambient_noise_rmsd(goal_xtc, noize_file, goal_prot_only, mul=0.8):
#     dist_arr = get_knn_dist_mdsctk(goal_xtc, noize_file, goal_prot_only)
#     min_rmsd = min(dist_arr)*mul  # I expect that current min does not represent real min.
#     print('Min rmsd for simulation is going to be : ', min_rmsd)
#     return min_rmsd
#
#
# def get_ambient_noise_angles(num_el, gro_file, noize_file, goal_bb_ndx, goal_angles, mul=0.8):
#     # generate filename
#     # convert_gro_to_xtc(gro_file, goal_bb_ndx)
#     sincos_file = 'noise_sincos.dat'
#     noize_file_bb = 'noize_bb.xtc'
#     angle_file = 'noise_angle.dat'
#
#     gmx_trjconv(f=noize_file, o=noize_file_bb, n=goal_bb_ndx, s=gro_file)
#     get_bb_to_angle_mdsctk(x=noize_file_bb, o=angle_file)
#     get_angle_to_sincos_mdsctk(i=angle_file, o=sincos_file)
#
#     os.remove(angle_file)
#
#     with open(sincos_file, 'rb') as file:
#         initial_1d_array = np.frombuffer(file.read(), dtype=np.float64, count=-1)
#     check_arr = np.reshape(initial_1d_array, (-1, num_el*2))
#     del initial_1d_array
#
#     res_arr = [None]*check_arr.shape[0]
#     for i in range(check_arr.shape[0]):
#         res_arr[i] = np.sum(abs(check_arr[i] - goal_angles))
#     return float(np.min(res_arr)*mul)


def compute_phipsi_angles(angl_num: int, target_filename: str, ndx: str, stor_name: str = None) -> np.ndarray:
    """Top level function that outputs sin/cos of the dihedral angles of the provided conformation.

    Args:
        :param int angl_num: total number of angles in the protein
        :param str target_filename:
        :param str ndx: index file to extract only specific atoms (extract the backbone)
        :param str stor_name:

    Returns:
        :return: array with sin/cos values of the backbone angles.
        :rtype: np.ndarray
    """
    xtc_filename = "{}.xtc".format(target_filename)
    if stor_name is None:  # then create temp file in /dev/shm
        bb_filename = "{}_bb.xtc".format(target_filename)
        ang_filename = "{}_bb.ang".format(target_filename)
        sin_cos_filename = "{}_bb.sc".format(target_filename)
        # making sure that we do not reuse old files
        if os.path.exists(bb_filename):
            os.remove(bb_filename)
        if os.path.exists(bb_filename):
            os.remove(bb_filename)
    else:  # then store in ./past/
        bb_filename = "{}_bb.xtc".format(stor_name)
        ang_filename = "{}_bb.ang".format(stor_name)
        sin_cos_filename = "{}_bb.sc".format(stor_name)

    gmx_trjconv(f=xtc_filename, o=bb_filename, n=ndx)
    get_bb_to_angle_mdsctk(x=bb_filename, o=ang_filename)
    get_angle_to_sincos_mdsctk(i=ang_filename, o=sin_cos_filename)

    with open(sin_cos_filename, 'rb') as file:
        initial_1d_array = np.frombuffer(file.read(), dtype=np.float64, count=-1)
    check_arr = np.reshape(initial_1d_array, (-1, angl_num * 2))
    if len(check_arr) == 1:
        return check_arr[0]
    return check_arr


def ang_dist(target_ang: list, goal_ang: list) -> np.ndarray:
    """Computes difference between two angle lists.

    Args:
        :param list target_ang: angles to test
        :param list goal_ang: goal angles

    Returns:
        :return: one number when input is a list or list of sums in case intput is list of lists
        :rtype: np.ndarray
    """
    if target_ang.shape[0] == 1 or target_ang.ndim == 1:
        return np.abs(target_ang - goal_ang).sum()
    else:
        return [np.abs(target_ang[i] - goal_ang).sum() for i in range(target_ang.shape[0])]


# def get_ambient_noise_contacts_xor(goal_prot_only, noize_xtc, ndx_file_cont, atom_num, logic_fun,
# corr_contacts, cont_dist, prev_cont, mult=0.8):
#     cont_sum, nat_contacts = get_native_contacts(goal_prot_only, [noize_xtc], ndx_file_cont,
# corr_contacts, atom_num, dist=cont_dist, logic_fun=logic_fun)
#     return max(1,int(min(abs(prev_cont - cont_sum))*mult))

# def get_ambient_noise_contacts(goal_prot_only, noize_xtc, ndx_file_cont, atom_num, logic_fun,
# corr_contacts, cont_dist, prev_cont, mult=0.8):
#     cont_sum, nat_contacts = get_native_contacts(goal_prot_only, [noize_xtc], ndx_file_cont,
# corr_contacts, atom_num, dist=cont_dist, logic_fun=logic_fun)
#     return max(1, int(min(abs(prev_cont - cont_sum)) * mult))


def save_an_file(an_file_name: str, tol_error: dict, metr_order: list) -> NoReturn:
    """Writes noise values into the specified file for future use during the restarts

    Args:
        :param str an_file_name: ambient noise filename
        :param dict tol_error: dict with ambient noise values for each metric
        :param list metr_order: list of metrics used in the current run

    Returns:
    Generates a file with noise values.
    """
    with open(an_file_name, 'w') as f:
        for metr_name in metr_order:
            f.write('{}\n'.format(tol_error[metr_name]))


def get_native_contacts(goal_prot_only: str, files_to_check: list, ndx_file: str, cont_corr: np.ndarray, atom_num: int,
                        dist: float = 2.7, logic_fun: np.ufunc = np.logical_xor, h_filter: list = None,
                        pool: mp.Pool = None, just_contacts: bool = False) -> tuple:  # goal_prot_only, files_for_trjcat, ndx_file
    """Computes number of contacts between the goal_prot_only and files_to_check.

    If files to check is a single list of contacts, then function returns int and list
    Otherwise it returns list of ints and list of lists

    Args:
        :param str goal_prot_only: .gro filename with stripped waters and salt
        :param list files_to_check: .xtc filename with frames we want to measure number of contacts with the goal
        :param str ndx_file: .ndx - index filename to select protein only in .xtc
        :param np.ndarray cont_corr: correct contacts between goal and goal (no mistakes) to compare with the files_to_check
        :param int atom_num: number of atoms used for memory (structure) allocation
        :param dist: distance that defines a contact
        :param np.ufunc logic_fun: defines what relation between the goal and the files_to_check we want to measure - AND, XOR
        :type logic_fun: Numpy logic function, typically logical_xor or logical_and
        :param list h_filter: boolean array with 1s in positions of H atoms, used to filter the final contacts
        :param mp.Pool pool: CPU pool - passed, since each instance does not deallocate the RAM
        :param bool just_contacts: flags to skip computation of the sum of correct contacts

    Returns:
        :return: sum of the correct contacts and contacts.
        :rtype: tuple
    """
    # nat_cont_arr = list()
    # contacts = list()
    if len(files_to_check) == 0:
        return None
    elif len(files_to_check) > 1:  # case for many files with one frame
        if pool is None:
            # pool = mp.Pool(mp.cpu_count())  # creation pool every time creates memory leak on python3.6.6 compiled with gcc 8.2.0
            raise Exception('Please pass pool variable')
        # ind = [get_contat_profile_mdsctk(goal_prot_only, file, ndx_file, dist)[1:] for file in files_to_check]
        ind = [elem[1:] for elem in pool.starmap(get_contat_profile_mdsctk,
                                                 ((goal_prot_only, file, ndx_file, dist) for file in files_to_check))]
        # corr_len = [elem[:1] for elem in ind if len(elem) > 0]
        contacts = [None] * len(ind)
        for i in range(len(ind)):
            elem = np.zeros(atom_num * atom_num, dtype=np.bool)
            elem[ind[i]] = True
            contacts[i] = elem
        del ind, elem, i
    else:  # case for one file with any number of frames
        cont_arr = get_contat_profile_mdsctk(goal_prot_only, files_to_check[0], ndx_file, dist)
        # print('Done with cont prof')
        if cont_arr[0] + 1 == len(cont_arr):  # we have only one frame
            full_arr = np.zeros(atom_num * atom_num, dtype=np.bool)
            full_arr[cont_arr[1:]] = True
            contacts = [full_arr]
            del full_arr
        else:  # we have many frames
            tot_ind = 0
            contacts = list()
            while tot_ind < len(cont_arr):
                tot_ind += 1
                next_ind = tot_ind + cont_arr[tot_ind - 1]
                full_arr = np.zeros(atom_num * atom_num, dtype=np.bool)
                full_arr[cont_arr[tot_ind:next_ind]] = True
                contacts.append(full_arr)
                tot_ind += cont_arr[tot_ind - 1]
            del cont_arr, tot_ind, next_ind, full_arr
    if not just_contacts:
        if h_filter is not None:
            contacts = [np.logical_and(arr_elem, h_filter) for arr_elem in contacts]  # while here we can just use logic_fun,
            # since we use filter only with AND to compute AND_H, I took a safe path
        nat_cont_sum_arr = [logic_fun(arr_elem, cont_corr).sum() for arr_elem in contacts]
    else:
        nat_cont_sum_arr = [None] * len(contacts)

    if len(nat_cont_sum_arr) == 1:
        return nat_cont_sum_arr[0], contacts[0]
    return nat_cont_sum_arr, contacts


def and_h(q: mp.Queue, goal_contacts_and_h_sum: np.int, goal_cont_h: list, contacts_h: list, prev_contacts_h: list, and_h_dist_tot: np.int) -> NoReturn:
    """Separate AND_H computation, used to be executed in parallel,

    NOT used anymore since does not result in any significant speed up, but left here "just in case".

    Args:
        :param mp.Queue q: queue used to communicate with the parent process
        :param np.int goal_contacts_and_h_sum: exact number of NMR contacts
        :param list goal_cont_h: correct (NMR) contacts
        :param list contacts_h: current nodes' contacts
        :param list prev_contacts_h: previous node contacts
        :param np.int and_h_dist_tot: distance accumulated from the origin

    Returns:
        :return: Returns by putting into the queue (metric to goal, metric from previous, total traveled in metric units).
    """
    goal_cont_dist_and_h = goal_contacts_and_h_sum - [np.logical_and(arr_elem, goal_cont_h).sum() for arr_elem in contacts_h]
    prev_cont_dist_and_h_1 = [np.logical_xor(arr_elem, prev_contacts_h).sum() for arr_elem in contacts_h]
    prev_cont_dist_and_h_2 = [arr_elem.sum() for arr_elem in contacts_h] + prev_contacts_h.sum()
    prev_cont_dist_and_h_2 = prev_cont_dist_and_h_2 / 2 - \
        [elem.sum() for elem in [np.logical_and(arr_elem, prev_contacts_h) for arr_elem in contacts_h]]
    total_cont_dist_and_h = and_h_dist_tot + prev_cont_dist_and_h_1
    q.put((goal_cont_dist_and_h, prev_cont_dist_and_h_2, total_cont_dist_and_h))


def and_p(q: mp.Queue, goal_contacts_and_sum: np.int, goal_contacts: list, contacts: list, prev_contacts: list, prev_tot_dist: np.int) -> NoReturn:
    """Separate AND computation, used to be executed in parallel,

    NOT used anymore since does not result in any significant speed up, but left here "just in case".

    Args:
        :param mp.Queue q: queue used to communicate with the parent process
        :param np.int goal_contacts_and_sum: exact number of NMR contacts
        :param list goal_contacts: correct (NMR) contacts
        :param list contacts: current nodes' contacts
        :param list prev_contacts: previous node contacts
        :param np.int prev_tot_dist: distance accumulated from the origin

    Returns:
        :return: Returns by putting into the queue (metric to goal, metric from previous, total traveled in metric units).
    """
    goal_cont_dist_and = goal_contacts_and_sum - [np.logical_and(arr_elem, goal_contacts).sum() for arr_elem in contacts]
    prev_cont_dist_and_1 = [np.logical_xor(arr_elem, prev_contacts).sum() for arr_elem in contacts]
    prev_cont_dist_and_2 = [arr_elem.sum() for arr_elem in contacts] + prev_contacts.sum()
    prev_cont_dist_and_2 = prev_cont_dist_and_2 / 2 - \
        [elem.sum() for elem in [np.logical_and(arr_elem, prev_contacts) for arr_elem in contacts]]
    total_cont_dist_and = prev_tot_dist + prev_cont_dist_and_1
    q.put((goal_cont_dist_and, prev_cont_dist_and_2, total_cont_dist_and))


def rmsd(q: mp.Queue, combined_pg: str, temp_xtc_file: str, goal_prot_only: str, prev_tot_dist: np.float64) -> NoReturn:
    """Separate RMSD computation, used to be executed in parallel,

    NOT used anymore since does not result in any significant speed up, but left here "just in case".

    Args:
        :param mp.Queue q: queue used to communicate with the parent process
        :param str combined_pg: two frames previous and goal
        :param str temp_xtc_file: new frames (same as number of seeds) you want to measure distance from previous and to the goal
        :param str goal_prot_only: goal protein only conformation
        :param np.float64 rev_tot_dist: distance accumulated from the origin

    Returns:
        :return: Returns by putting into the queue (metric to goal, metric from previous, total traveled in metric units).
    """
    dist_arr = get_knn_dist_mdsctk(combined_pg, temp_xtc_file, goal_prot_only)
    from_prev_dist = dist_arr[0::2]
    rmsd_to_goal = dist_arr[1::2]
    rmsd_total_trav = [prev_tot_dist + elem for elem in from_prev_dist]
    q.put((rmsd_to_goal, from_prev_dist, rmsd_total_trav))


def angl(q: mp.Queue, angl_num: int, temp_xtc_file: str, init_bb_ndx: str, pangl: list, goal_angles: list, prev_tot_dist: np.float64) -> NoReturn:
    """Separate ANGL computation, used to be executed in parallel,

    NOT used anymore since does not result in any significant speed up, but left here "just in case".

    Args:
        :param mp.Queue q: queue used to communicate with the parent process
        :param int angl_num: total number of angles in the protein
        :param str temp_xtc_file: new frames (same as number of seeds) you want to measure distance from previous and to the goal
        :param str init_bb_ndx: .ndx to extract the backbone atoms
        :param list pangl: previous node angles
        :param list goal_angles: correct angles (NMR angles)
        :param np.float64 prev_tot_dist: distance accumulated from the origin

    Returns:
        :return: Returns by putting into the queue (metric to goal, metric from previous, total traveled in metric units).
    """
    cur_angles = compute_phipsi_angles(angl_num, temp_xtc_file.split('.')[0], init_bb_ndx)
    angl_sum_from_prev = ang_dist(cur_angles, pangl)
    angl_sum_to_goal = ang_dist(cur_angles, goal_angles)
    angl_sum_tot = prev_tot_dist + angl_sum_from_prev
    q.put((angl_sum_to_goal, angl_sum_from_prev, angl_sum_tot, cur_angles))


def compute_metric(past_dir: str, new_nodes_names: list, tot_seeds: int, combined_pg: str, temp_xtc_file: str, goal_prot_only: str, node_info: dict, angl_num: int,
                   init_bb_ndx: str, goal_angles: list, init_prot_only: str, files_for_trjcat: list, ndx_file_init: str, goal_cont_h: list, atom_num: int,
                   cont_dist: float, h_filter_init: list, goal_contacts: list, cur_metric: int, goal_contacts_and_h_sum: np.int, goal_contacts_and_sum: np.int,
                   chance_to_reuse: bool = False, cpu_pool: mp.Pool = None, compute_all_at_once: bool = True) -> list:
    """Computes metric distances from the previous node and to the goal (NMR) conformation.

    Before I was computing metrics separately, but computing them all at once add very little overhead
     and allows to track trajectory behavior, so later I fixed only the code with all at once option.

    Args:
        :param str past_dir: path to the directory with prior computation results
        :param list new_nodes_names: full names of newly computed nodes (not current)
        :param int tot_seeds: total number of seed in the current run
        :param str combined_pg: previous and goal frames combined into one trajectory
        :param str temp_xtc_file: new nodes' final frames
        :param str goal_prot_only: NMR (folded) conformation without water and salt (protein only)
        :param dict node_info: info about the current node (not just computed, but rather previous)
        :param int angl_num: number of dihedral angles in the protein
        :param str init_bb_ndx: index file with backbone atom positions for the initial conformation
        :param list goal_angles: angle values of the NMR structure
        :param str init_prot_only: initial (unfolded) conformation without water and salt (protein only)
        :param list files_for_trjcat: list of newly computed nodes (files, with hash as a name)
        :param str ndx_file_init: index file with backbone atom positions for the NMR conformation
        :param list goal_cont_h: contact values of the NMR structure (hydrogens only)
        :param int atom_num: total number of atoms in the protein (same for folded and unfolded)
        :param float cont_dist: distance between atoms treated as 'contact'
        :param list h_filter_init: positions of the hydrogen atoms in the initial (unfolded) conformation
        :param list goal_contacts: list of correct contacts in the NMR (folded) conformation
        :param int cur_metric: metric index
        :param np.int goal_contacts_and_h_sum: total sum of the contacts between hydrogents in the NMR (folded) conformation
        :param np.int goal_contacts_and_sum: total sum of the contacts in the NMR (folded) conformation
        :param bool chance_to_reuse:
        :param mp.Pool cpu_pool: CPU pool for local parallel processing
        :param bool compute_all_at_once: toggle whether to compute all metrics at the same time or not (yes, if no check the code)

    Returns:
        :return: new nodes with all metrics (compute_all_at_once only) and current metric distances
        :rtype: list
    """
    # global extra_past
    new_nodes = [None] * tot_seeds
    # prev_contacts = node_info['contacts']
    try:
        prev_contacts = load_npz(os.path.join(past_dir, '{}.cont.npz'.format(node_info['digest_name']))).toarray()
    except:
        print('Previous contact do not exists. Probably error in the previous step.\nFile: ',
              os.path.join(past_dir, '{}.cont.npz'.format(node_info['digest_name'])),
              ' was not found')
        exit(-10)
        # prev_contacts = load_npz(os.path.join(extra_past, '{}.cont.npz'.format(node_info['digest_name']))).toarray()
    digests = [get_digest(new_nodes_names[i]) for i in range(tot_seeds)]
    if compute_all_at_once:
        # Parallel approach does not work on small/medium proteins. Overhead of proc creation is more than time to compute.
        # However, when you decide to speed up execution, make only angl dist to be computed in sep process.
        # q = mp.Queue()
        # pid = multiprocessing.Process(target=angl, args=(q, angl_num, temp_xtc_file, init_bb_ndx, node_info['angles'],
        # goal_angles, node_info['ANGL_dist_total']))
        # pid.start()

        # *********  PREP ************
        reusing_old_cont = False
        # if chance_to_reuse:
        try:  # lets always check for previous files and regenerate them in case of the error - incomplete or do not exist
            contacts = [load_npz(os.path.join(past_dir, '{}.cont.npz'.format(digests[i]))).toarray() for i in range(tot_seeds)]
            reusing_old_cont = True
        except OSError:
            contacts = get_native_contacts(init_prot_only, files_for_trjcat, ndx_file_init, None,
                                           atom_num, cont_dist, None, pool=cpu_pool, just_contacts=True)[1]
        # else:
        #     contacts = get_native_contacts(init_prot_only, files_for_trjcat, ndx_file_init, None,
        #                                    atom_num, cont_dist, None, pool=cpu_pool, just_contacts=True)[1]

        # print(init_prot_only, files_for_trjcat, ndx_file_init, atom_num, cont_dist)
        #  Cont prep
        contacts_h = [np.logical_and(arr_elem, h_filter_init) for arr_elem in contacts]
        prev_contacts_h = np.logical_and(prev_contacts, h_filter_init)

        # ************** PAR ************
        # q = [mp.Queue() for i in range(4)]
        # bad approach
        # par_metr = [multiprocessing.Process(target=and_h, args=(q[0], goal_contacts_and_h_sum, goal_cont_h, contacts_h,
        # prev_contacts_h, node_info['AND_H_dist_total'])),
        #             multiprocessing.Process(target=and_p, args=(q[1], goal_contacts_and_sum, goal_contacts, contacts,
        # prev_contacts, node_info['AND_dist_total'])),
        #             multiprocessing.Process(target=rmsd, args=(q[2], combined_pg, temp_xtc_file,
        # goal_prot_only, node_info['RMSD_dist_total'])),
        #             multiprocessing.Process(target=angl, args=(q[3], angl_num, temp_xtc_file, init_bb_ndx,
        # node_info['angles'], goal_angles, node_info['ANGL_dist_total']))]
        # [pid.start() for pid in par_metr]
        # [pid.join() for pid in par_metr]
        # goal_cont_dist_and_h, prev_cont_dist_and_h_2, total_cont_dist_and_h = q[0].get()
        # goal_cont_dist_and, prev_cont_dist_and_2, total_cont_dist_and = q[1].get()
        # rmsd_to_goal, from_prev_dist, rmsd_total_trav = q[2].get()
        # angl_sum_to_goal, angl_sum_from_prev, angl_sum_tot, cur_angles = q[3].get()
        #
        # better approach
        # q = [mp.Queue() for i in range(4)]
        # pid = multiprocessing.Process(target=angl, args=(q[3], angl_num, temp_xtc_file, init_bb_ndx, node_info['angles'],
        # goal_angles, node_info['ANGL_dist_total']))
        # pid.start()
        # and_h(q[0], goal_contacts_and_h_sum, goal_cont_h, contacts_h, prev_contacts_h, node_info['AND_H_dist_total'])
        # and_p(q[1], goal_contacts_and_sum, goal_contacts, contacts, prev_contacts, node_info['AND_dist_total'])
        # rmsd(q[2], combined_pg, temp_xtc_file, goal_prot_only, node_info['RMSD_dist_total'])
        # pid.join()
        # angl_sum_to_goal, angl_sum_from_prev, angl_sum_tot, cur_angles = q[3].get()

        # *********  RMSD ************
        dist_arr = get_knn_dist_mdsctk(combined_pg, temp_xtc_file, goal_prot_only)
        from_prev_dist = dist_arr[0::2]
        rmsd_to_goal = dist_arr[1::2]
        rmsd_total_trav = [node_info['RMSD_dist_total'] + elem for elem in from_prev_dist]

        # *********  ANG ************
        reusing_old_angl = False
        # if chance_to_reuse:
        try:
            cur_angles = [np.fromfile(os.path.join(past_dir, '{}.angl'.format(digests[i])), dtype=np.float32) for i in range(tot_seeds)]
            cur_angles = np.asarray(cur_angles, dtype=np.float32)
            reusing_old_angl = True
        except OSError:
            cur_angles = compute_phipsi_angles(angl_num, temp_xtc_file.split('.')[0], init_bb_ndx)
        # else:
        #     cur_angles = compute_phipsi_angles(angl_num, temp_xtc_file.split('.')[0], init_bb_ndx)

        # angl_sum_from_prev = ang_dist(cur_angles, node_info['angles'])
        # if os.path.exists(os.path.join(past_dir, '{}.angl'.format(node_info['digest_name']))):
        try:
            angl_sum_from_prev = ang_dist(cur_angles, np.fromfile(os.path.join(past_dir, '{}.angl'.format(node_info['digest_name'])), dtype=np.float32))
        except Exception as e:
            print('Error during previous angle read.\nCheck ', os.path.join(past_dir, '{}.angl'.format(node_info['digest_name'])), 'Error: ', e)
            exit(-10)
        # else:
            # angl_sum_from_prev = ang_dist(cur_angles, np.fromfile(os.path.join(extra_past, '{}.angl'.format(node_info['digest_name'])), dtype=np.float32))
        angl_sum_to_goal = ang_dist(cur_angles, goal_angles)
        angl_sum_tot = node_info['ANGL_dist_total'] + angl_sum_from_prev

        # *********  AND_H ************
        goal_cont_dist_and_h = goal_contacts_and_h_sum - [np.logical_and(arr_elem, goal_cont_h).sum() for arr_elem in contacts_h]
        prev_cont_dist_and_h_1 = [np.logical_xor(arr_elem, prev_contacts_h).sum() for arr_elem in contacts_h]
        # prev_cont_dist_and_h_2 = [arr_elem.sum() for arr_elem in contacts_h] + prev_contacts_h.sum()
        # prev_cont_dist_and_h_2 = prev_cont_dist_and_h_2 / 2 - \
        #    [elem.sum() for elem in [np.logical_and(arr_elem, prev_contacts_h) for arr_elem in contacts_h]]
        total_cont_dist_and_h = node_info['AND_H_dist_total'] + prev_cont_dist_and_h_1

        # *********  AND ************
        goal_cont_dist_and = goal_contacts_and_sum - [np.logical_and(arr_elem, goal_contacts).sum() for arr_elem in contacts]
        prev_cont_dist_and_1 = [np.logical_xor(arr_elem, prev_contacts).sum() for arr_elem in contacts]
        # prev_cont_dist_and_2 = [arr_elem.sum() for arr_elem in contacts] + prev_contacts.sum()
        # prev_cont_dist_and_2 = prev_cont_dist_and_2 / 2 - \
        #                      [elem.sum() for elem in [np.logical_and(arr_elem, prev_contacts) for arr_elem in contacts]]
        total_cont_dist_and = node_info['AND_dist_total'] + prev_cont_dist_and_1

        # *********  XOR ************
        goal_cont_dist_sum_xor = [np.logical_xor(arr_elem, goal_contacts).sum() for arr_elem in contacts]
        # prev_cont_dist_sum_xor = [np.logical_xor(arr_elem, prev_contacts).sum() for arr_elem in contacts]
        prev_cont_dist_sum_xor = prev_cont_dist_and_1  # it is the same, no need to compute twice
        total_cont_dist_xor = node_info['XOR_dist_total'] + prev_cont_dist_sum_xor

        # # END PAR
        # pid.join()
        # angl_sum_to_goal, angl_sum_from_prev, angl_sum_tot, cur_angles = q.get()

        # store all metrics
        for i in range(tot_seeds):
            new_nodes[i] = dict()
            new_nodes[i]['digest_name'] = get_digest(new_nodes_names[i])

            new_nodes[i]['RMSD_to_goal'] = np.float32(rmsd_to_goal[i])
            new_nodes[i]['RMSD_from_prev'] = np.float32(from_prev_dist[i])
            new_nodes[i]['RMSD_dist_total'] = np.float32(rmsd_total_trav[i])

            new_nodes[i]['ANGL_to_goal'] = np.float32(angl_sum_to_goal[i])
            new_nodes[i]['ANGL_from_prev'] = np.float32(angl_sum_from_prev[i])
            new_nodes[i]['ANGL_dist_total'] = np.float32(angl_sum_tot[i])

            new_nodes[i]['AND_H_to_goal'] = np.int32(goal_cont_dist_and_h[i])
            new_nodes[i]['AND_H_from_prev'] = np.int32(prev_cont_dist_and_h_1[i])
            new_nodes[i]['AND_H_dist_total'] = np.int32(total_cont_dist_and_h[i])

            new_nodes[i]['AND_to_goal'] = np.int32(goal_cont_dist_and[i])
            new_nodes[i]['AND_from_prev'] = np.int32(prev_cont_dist_and_1[i])
            new_nodes[i]['AND_dist_total'] = np.int32(total_cont_dist_and[i])

            new_nodes[i]['XOR_to_goal'] = np.int32(goal_cont_dist_sum_xor[i])
            new_nodes[i]['XOR_from_prev'] = np.int32(prev_cont_dist_sum_xor[i])
            new_nodes[i]['XOR_dist_total'] = np.int32(total_cont_dist_xor[i])

            new_nodes[i]['native_name'] = zlib.compress(new_nodes_names[i].encode(), 9)
            # new_nodes[i]['contacts'] = csc_matrix(contacts[i])  # csc is the most efficient for contacts data, I tested it.
            # new_nodes[i]['angles']   = cur_angles[i].astype('float32')

            if not reusing_old_cont:
                save_npz((os.path.join(past_dir, '{}.cont'.format(new_nodes[i]['digest_name']))), csc_matrix(contacts[i]), compressed=True)

            if not reusing_old_angl:
                cur_angles[i].astype('float32').tofile(os.path.join(past_dir, '{}.angl'.format(new_nodes[i]['digest_name'])))

        if cur_metric == 0:
            return new_nodes, rmsd_to_goal, from_prev_dist, rmsd_total_trav
        elif cur_metric == 1:
            return new_nodes, angl_sum_to_goal, angl_sum_from_prev, angl_sum_tot
        elif cur_metric == 2:
            # if not isinstance(goal_cont_dist_and_h, (list,)):
            #     raise Exception('AND_H_to_goal: ', goal_cont_dist_and_h)
            return new_nodes, list(goal_cont_dist_and_h), list(prev_cont_dist_and_h_1), list(total_cont_dist_and_h)
        elif cur_metric == 3:
            # if not isinstance(goal_cont_dist_and, (list,)):
            #     raise Exception('AND_to_goal: ', goal_cont_dist_and)
            return new_nodes, list(goal_cont_dist_and), list(prev_cont_dist_and_1), list(total_cont_dist_and)
        elif cur_metric == 4:
            # if not isinstance(goal_cont_dist_sum_xor, (list,)):
            #     raise Exception('XOR_to_goal: ', goal_cont_dist_sum_xor)
            return new_nodes, list(goal_cont_dist_sum_xor), list(prev_cont_dist_sum_xor), list(total_cont_dist_xor)
        else:
            raise Exception('Unknown metric')
    else:  # This version is outdated. Using one metric does not produce significant speedup
        if cur_metric == 0:  # RMSD
            dist_arr = get_knn_dist_mdsctk(combined_pg, temp_xtc_file, goal_prot_only)
            # TODO: fix rm files and check if other files has to be removed
            # rm_queue.put_nowait(combined_pg)
            # rm_queue.put_nowait(temp_xtc_file)
            # since combined_pg had two points we have to divide result into two arrays
            from_prev_dist = dist_arr[0::2]
            rmsd_to_goal = dist_arr[1::2]
            rmsd_total_trav = [node_info['RMSD_dist_total'] + elem for elem in from_prev_dist]
            for i in range(tot_seeds):
                new_nodes[i]['RMSD_to_goal'] = rmsd_to_goal[i]
                new_nodes[i]['RMSD_from_prev'] = from_prev_dist[i]
                new_nodes[i]['RMSD_dist_total'] = rmsd_total_trav[i]

            return new_nodes, rmsd_to_goal, from_prev_dist, rmsd_total_trav

        elif cur_metric == 1:  # PhyPsi
            cur_angles = compute_phipsi_angles(angl_num, temp_xtc_file.split('.')[0], init_bb_ndx)
            angl_sum_from_prev = ang_dist(cur_angles, node_info['angles'])
            angl_sum_to_goal = ang_dist(cur_angles, goal_angles)
            angl_sum_tot = node_info['ANG_dist_total'] + angl_sum_from_prev
            for i in range(tot_seeds):
                new_nodes[i]['ANGL_to_goal'] = angl_sum_to_goal[i]
                new_nodes[i]['ANGL_from_prev'] = angl_sum_from_prev[i]
                new_nodes[i]['ANGL_dist_total'] = angl_sum_tot[i]
                new_nodes[i]['angles'] = cur_angles[i]

            return new_nodes, angl_sum_to_goal, angl_sum_from_prev, angl_sum_tot

        elif cur_metric == 2:  # AND_H
            contacts = get_native_contacts(init_prot_only, files_for_trjcat, ndx_file_init, goal_contacts,
                                           atom_num, cont_dist, np.logical_and, pool=cpu_pool)[1]
            # although it is possible to get h_contacts from the get_native_contacts, then I'll not be able to get pure contacts to store
            contacts_h = [np.logical_and(arr_elem, h_filter_init) for arr_elem in contacts]
            goal_cont_dist_and_h = [np.logical_and(arr_elem, goal_cont_h).sum() for arr_elem in contacts_h]
            prev_contacts_h = np.logical_and(prev_contacts.toarray(), h_filter_init)
            prev_cont_dist_and_h_1 = [np.logical_xor(arr_elem, prev_contacts_h).sum() for arr_elem in contacts_h]
            prev_cont_dist_and_h_2 = [arr_elem.sum() for arr_elem in contacts_h] + prev_contacts_h.sum()
            prev_cont_dist_and_h_2 = prev_cont_dist_and_h_2 / 2 - \
                [elem.sum() for elem in [np.logical_and(arr_elem, prev_contacts_h) for arr_elem in contacts_h]]
            total_cont_dist_and_h = node_info['AND_H_dist_total'] + prev_cont_dist_and_h_1
            for i in range(tot_seeds):
                new_nodes[i]['AND_H_to_goal'] = goal_cont_dist_and_h[i]
                new_nodes[i]['AND_H_from_prev'] = prev_cont_dist_and_h_1[i]
                new_nodes[i]['AND_H_dist_total'] = total_cont_dist_and_h[i]
                new_nodes[i]['contacts'] = csc_matrix(contacts[i])

            return new_nodes, goal_cont_dist_and_h, prev_cont_dist_and_h_1, total_cont_dist_and_h

        elif cur_metric == 3:  # AND
            goal_cont_dist_and, contacts = get_native_contacts(init_prot_only, files_for_trjcat, ndx_file_init, goal_contacts,
                                                               atom_num, cont_dist, np.logical_and, pool=cpu_pool)
            prev_cont_dist_and_1 = [np.logical_xor(arr_elem, prev_contacts.toarray()).sum() for arr_elem in contacts]
            prev_cont_dist_and_2 = [arr_elem.sum() for arr_elem in contacts] + prev_contacts.sum()
            prev_cont_dist_and_2 = prev_cont_dist_and_2 / 2 - \
                [elem.sum() for elem in [np.logical_and(arr_elem, prev_contacts.toarray()) for arr_elem in contacts]]
            total_cont_dist_and = node_info['AND_dist_total'] + prev_cont_dist_and_1
            for i in range(tot_seeds):
                new_nodes[i]['AND_to_goal'] = goal_cont_dist_and[i]
                new_nodes[i]['AND_from_prev'] = prev_cont_dist_and_1[i]
                new_nodes[i]['AND_dist_total'] = total_cont_dist_and[i]
                new_nodes[i]['contacts'] = csc_matrix(contacts[i])

            return new_nodes, goal_cont_dist_and, prev_cont_dist_and_1, total_cont_dist_and

        elif cur_metric == 4:  # XOR
            goal_cont_dist_xor, contacts = get_native_contacts(init_prot_only, files_for_trjcat, ndx_file_init, goal_contacts,
                                                               atom_num, cont_dist, np.logical_xor, pool=cpu_pool)
            prev_cont_dist_sum_xor = [np.logical_xor(arr_elem, prev_contacts.toarray()).sum() for arr_elem in contacts]
            total_cont_dist_xor = node_info['XOR_dist_total'] + prev_cont_dist_sum_xor
            for i in range(tot_seeds):
                new_nodes[i]['XOR_to_goal'] = goal_cont_dist_xor[i]
                new_nodes[i]['XOR_from_prev'] = prev_cont_dist_sum_xor[i]
                new_nodes[i]['XOR_dist_total'] = total_cont_dist_xor[i]
                new_nodes[i]['contacts'] = csc_matrix(contacts[i])

            return new_nodes, goal_cont_dist_xor, prev_cont_dist_sum_xor, total_cont_dist_xor

    raise Exception("You cant be here")


def compute_init_metric(past_dir: str, tot_seeds: int, init_xtc: str, goal_xtc: str, goal_prot_only: str, angl_num: int,
                        init_bb_ndx: str, goal_angles: np.ndarray, init_prot_only: str, ndx_file_init: str,
                        goal_cont_h: np.ndarray, atom_num: int, cont_dist: float, h_filter_init: np.ndarray,
                        goal_contacts: np.ndarray, goal_contacts_and_h_sum: np.int64, goal_contacts_and_sum: np.int64) -> list:
    """Special case of the "compute_metric"

    Computes metric distances to the goal (NMR) conformation and sets previous distances to 0

    Args:
        :param str past_dir: path to the directory with prior computation results
        :param int tot_seeds: total number of seed in the current run
        :param str init_xtc: initial (unfolded) conformation with water and salt
        :paramstr  goal_xtc: NMR (folded) conformation with water and salt
        :param str goal_prot_only: NMR (folded) conformation without water and salt (protein only)
        :param int angl_num: number of dihedral angles in the protein
        :param str init_bb_ndx: index file with backbone atom positions for the initial conformation
        :param np.ndarray goal_angles: angle values of the NMR structure
        :param str init_prot_only: initial (unfolded) conformation without water and salt (protein only)
        :param str ndx_file_init: index file with backbone atom positions for the NMR conformation
        :param np.ndarray goal_cont_h: contact values of the NMR structure (hydrogens only)
        :param int atom_num: total number of atoms in the protein (same for folded and unfolded)
        :param float cont_dist: distance between atoms treated as 'contact'
        :param np.ndarray h_filter_init: positions of the hydrogen atoms in the initial (unfolded) conformation
        :param np.ndarray goal_contacts: list of correct contacts in the NMR (folded) conformation
        :param np.int64 goal_contacts_and_h_sum: total sum of the contacts between hydrogents in the NMR (folded) conformation
        :param np.int64 goal_contacts_and_sum: total sum of the contacts in the NMR (folded) conformation

    Returns:
        :return: node structure with the initial metrics
        :rtype: list
    """
    init_node = [None] * tot_seeds
    dim = 1 if tot_seeds > 1 else 0
    # *********  RMSD ************
    rmsd_to_goal = get_knn_dist_mdsctk(init_xtc, goal_xtc, goal_prot_only)
    # *********  ANG ************
    cur_angles = compute_phipsi_angles(angl_num, init_xtc.split('.')[0], init_bb_ndx)

    angl_sum_to_goal = ang_dist(cur_angles, goal_angles)

    contacts = get_native_contacts(init_prot_only, [init_xtc], ndx_file_init, None, atom_num, cont_dist, None, just_contacts=True)[1]
    # print(init_prot_only, init_xtc, ndx_file_init, atom_num, cont_dist)
    #  Cont prep
    contacts_h = np.logical_and(contacts, h_filter_init)
    # *********  AND_H ************
    goal_cont_dist_and_h = goal_contacts_and_h_sum - np.logical_and(contacts_h, goal_cont_h).sum(axis=dim)
    # *********  AND ************
    goal_cont_dist_and = goal_contacts_and_sum - np.logical_and(contacts, goal_contacts).sum(axis=dim)
    # *********  XOR ************
    goal_cont_dist_sum_xor = np.logical_xor(contacts, goal_contacts).sum(axis=dim)

    if dim == 0:
        contacts = [contacts]
        # contacts_h = [contacts_h]
        angl_sum_to_goal = [angl_sum_to_goal]
        goal_cont_dist_and_h = [goal_cont_dist_and_h]
        goal_cont_dist_and = [goal_cont_dist_and]
        goal_cont_dist_sum_xor = [goal_cont_dist_sum_xor]

    # store all metrics
    for i in range(tot_seeds):
        init_node[i] = dict()
        init_node[i]['digest_name'] = get_digest('s')

        init_node[i]['RMSD_to_goal'] = np.float32(rmsd_to_goal[i])
        init_node[i]['RMSD_from_prev'] = np.uint32(0)
        init_node[i]['RMSD_dist_total'] = np.uint32(0)

        init_node[i]['ANGL_to_goal'] = np.float32(angl_sum_to_goal[i])
        init_node[i]['ANGL_from_prev'] = np.uint32(0)
        init_node[i]['ANGL_dist_total'] = np.uint32(0)

        init_node[i]['AND_H_to_goal'] = np.uint32(goal_cont_dist_and_h[i])
        init_node[i]['AND_H_from_prev'] = np.uint32(0)
        init_node[i]['AND_H_dist_total'] = np.uint32(0)

        init_node[i]['AND_to_goal'] = np.uint32(goal_cont_dist_and[i])
        init_node[i]['AND_from_prev'] = np.uint32(0)
        init_node[i]['AND_dist_total'] = np.uint32(0)

        init_node[i]['XOR_to_goal'] = np.uint32(goal_cont_dist_sum_xor[i])
        init_node[i]['XOR_from_prev'] = np.uint32(0)
        init_node[i]['XOR_dist_total'] = np.uint32(0)
        # init_node[i]['contacts'] = csc_matrix(contacts[i])
        save_npz(os.path.join(past_dir, '{}.cont'.format(init_node[i]['digest_name'])),
                 csc_matrix(contacts[i]), compressed=True)

        init_node[i]['native_name'] = zlib.compress('s'.encode(), 9)

        # init_node[i]['angles'] = cur_angles[i]
        cur_angles.astype('float32').tofile(os.path.join(past_dir, '{}.angl'.format(init_node[i]['digest_name'])))

    if len(init_node) == 1:
        return init_node[0]
    return init_node


def select_metrics_by_snr(cur_nodes: list, prev_node: dict, metric_names: list, tol_error: dict,
                          compute_all_at_once: bool, alowed_metrics: list, cur_metr: str) -> str:
    """SNR approach to a metric selection.

    Metrics that had the highest SNR ratio (metric distance from the prev point)/(ambient noise) is selected next
    However, this approach does not always work and while you may a high SNR with contacts, there may be no real decrease in the rmsd.
    It is affected by the previous point performance.

    Args:
        :param list cur_nodes: recent nodes
        :param dict prev_node: previous node
        :param list metric_names: list of metrics implemented (I want to know whole statistics, not only allowed metrics)
        :param dict tol_error: dict with noise data
        :param bool compute_all_at_once: toggle left as a reminder to not implement all at once
        :param list alowed_metrics: list of metrics that we allow to be used during the current run
        :param str cur_metr: name of the current metric

    Returns:
        :return: metric name with the highest SNR
    """
    if not compute_all_at_once:
        # easy to implement, but I do not have plans to use it since 'all at once' is very fast
        # just take last node and compute all metrics
        raise Exception('Not implemented')

    snr = False
    if snr:  # SNR approach may be biased. Additionally, prev_node should be computed here as prev point in name: s_1 is prev to s_1_3
        signal = dict()
        best_metr = metric_names[0]
        best_val = -1
        for metr in metric_names:
            cur_name = '{}_to_goal'.format(metr)
            signal[metr] = 0
            for i in range(len(cur_nodes)):
                signal[metr] += (cur_nodes[i][cur_name] - prev_node[cur_name]) / tol_error[metr]
            if metric_names != metric_names[0] and signal[metr] > best_val and metr in alowed_metrics:
                best_val = signal[metr]
                best_metr = metr

        if best_metr == cur_metr:
            print('New metric is the same as previous. Switching to next metric')
            while len(metric_names) > 1 and (best_metr == cur_metr or best_metr not in alowed_metrics):
                best_metr = metric_names[(metric_names.index(best_metr) + 1) % len(metric_names)]

        print('SNR for metrics:')
        for metr in metric_names:
            if metr == best_metr:
                print(' >*{}: {}'.format(metr, signal[metr]))
            elif best_val == signal[metr]:
                print('  +{}: {}'.format(metr, signal[metr]))
            elif metr not in alowed_metrics:
                print('   {}: {} # ignored'.format(metr, signal[metr]))
            else:
                print('   {}: {}'.format(metr, signal[metr]))
    else:  # use round-robin
        best_metr = metric_names[(metric_names.index(cur_metr) + 1) % len(metric_names)]
        while best_metr not in alowed_metrics:
            print('Skipping {} since it is not in allowed list'.format(best_metr))
            best_metr = metric_names[(metric_names.index(cur_metr) + 1) % len(metric_names)]
        print('Switching to {}'.format(best_metr))

    return best_metr
