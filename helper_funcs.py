import os
import multiprocessing
import hashlib
from shutil import copy2 as cp2
import heapq
import shutil
import pickle

from typing import NoReturn

from gen_mdp import get_mdp
from gmx_wrappers import gmx_grompp, gmx_trjconv, gmx_trjcat, gmx_mdrun, gmx_mdrun_mpi, gmx_mdrun_mpi_with_sched


def get_digest(in_str: str) -> str:
    """
    Computes digest of the input string.
    :param in_str: typically list of seeds concatenated with _. like s_0_1_5
    :return: blake2 hash of the in_str. We use short version, but you can use full version - slightly slower, but less chances of name collision.
    :rtype: str
    """
    # return hashlib.md5(in_str.encode()).hexdigest()
    # if you have python older than 3.6 - use md5 or update python
    return hashlib.blake2s(in_str.encode()).hexdigest()


def create_core_mapping(ncores=multiprocessing.cpu_count(), nseeds: int = 1) -> list:
    """
    Tries to map cores evenly among tasks.
    :param ncores: number of cores available
    :param nseeds: number of seeds used in current run
    :return: list of tupples, each tupple consist of (cores number, task identifier)
    :rtype: list
    """
    ncores = ncores if ncores > 0 else 1
    nseeds = nseeds if nseeds > 0 else 1
    print('I will use {} cores for {} seeds'.format(ncores, nseeds))

    even = ncores // nseeds
    remainder = ncores % nseeds

    sched_arr = list()
    if even:
        cur_sched = [(even+1, i) if i < remainder else (even, i) for i in range(nseeds)]
        sched_arr.append(cur_sched)
    else:
        seeds_range_iter = iter(range(nseeds))
        tot_batches = nseeds//ncores
        remainder = nseeds-tot_batches*ncores
        tot_batches = tot_batches if not remainder else tot_batches+1  # if we can`t divide tasks evenly, we need one more batch
        for i in range(tot_batches):
            if i < tot_batches-1:
                cur_sched = [(1, 0)]*ncores
            else:
                cur_sched = [(1, 0) if i < remainder else (0, 0) for i in range(ncores)]
                free_cores = ncores - sum(i for i, j in cur_sched)
                if free_cores:
                    cur_sched = [(j[0]+1, 0) if i < free_cores else (j[0], 0) for i, j in enumerate(cur_sched)]
            sched_arr.append(cur_sched)
        for i, cur_sched in enumerate(sched_arr):
            for j, cornum_seed in enumerate(cur_sched):
                if cornum_seed[0]:
                    cur_seed = next(seeds_range_iter)
                    sched_arr[i][j] = (cornum_seed[0], cur_seed)
                    print('Seed {} will be run on {} cores.'.format(cur_seed, cornum_seed[0]))

    return sched_arr


def get_previous_runs_info(check_dir: str) -> list:
    """
    Scans direcotory for prior results and outputs the list of filenames
    :param check_dir:  directory to scan for prior trajectories
    :return: list of filenames .xtc or .gro
    :rtype: list
    """
    # filenames_found = os.walk(check_dir).__next__()[2]
    filenames_found = [f.split("/")[-1] for f in os.listdir(check_dir)]
    # filenames_found = [f.path.split("/")[-1] for f in os.scandir(check_dir)]
    filenames_found_important = [f for f in filenames_found if f.split('.')[1] in ['xtc', 'gro']]
    del filenames_found
    print('Found files: {} with .gro and .xtc'.format(len(filenames_found_important)))
    return filenames_found_important


def check_precomputed_noize(an_file: str, metr_order: list):
    """
    Checks whether file with precomputed ambient noise exists.
    Tries to read correct number of metrics, in case of error throws and exception
    Otherwise returns dict{metric_name: noise_value}
    :param an_file: ambient noise filename to check
    :param metr_order: order of metric names (should be correct sequence)
    :return: dict{metric_name: noise_value}
    :rtype: dict or None
    """
    # TODO: rewrite function to save noise and metric name, so you do not read the wrong sequence (add a check)
    if an_file in os.walk(".").__next__()[2]:
        print(an_file, ' was found. Reading... ')
        with open(an_file, 'r') as f:
            noize_arr = f.readlines()
        try:
            res_arr = [float(res.strip()) for res in noize_arr]
            err_node = dict()
            for i in range(len(res_arr)):
                err_node[metr_order[i]] = res_arr[i]
        except Exception as e:
            print(e)
            return None
        return err_node
    return None


def make_a_step(work_dir: str, cur_seed: int, seed_dirs: dict, top_file: str, ndx_file: str, seed_digest_filename: str, old_name_digest: str, past_dir: str, ncores: int = 1) -> NoReturn:
    """
    Generates the actual MD simulation.
    :param work_dir: path to the directory where seed dirs reside
    :param cur_seed: current seed value used for MD production
    :param seed_dirs: dict which contains physical path to the directory where simulation with particular seed is performed
    :param top_file: .top - topology of the current conformation
    :param ndx_file: .ndx - index of the protein atoms of the current conformation
    :param seed_digest_filename: digest for a current MD simulation, used to store files in the past
    :param old_name_digest: digest for a prior MD simulation
    :param past_dir: path to the directory with prior computations
    :param ncores: number of cores to use for this task
    """
    # global extra_past
    old_name = os.path.join(past_dir, old_name_digest)
    if not os.path.exists(old_name+'.gro'):
        # old_name = os.path.join(extra_past, old_name_digest)
        # if not os.path.exists(old_name + '.gro'):
        raise Exception("make_a_step: did not find {} in {} ".format(old_name_digest, past_dir))
    gmx_grompp(work_dir, cur_seed, top_file, old_name)
    new_name = os.path.join(past_dir, seed_digest_filename)
    gmx_mdrun(work_dir, cur_seed, new_name + '.gro', ncores)
    gmx_trjconv(f=os.path.join(seed_dirs[cur_seed], 'md.xtc'), o='{}.xtc'.format(new_name),
                n=ndx_file, s=os.path.join(seed_dirs[cur_seed], 'md.tpr'), pbc='mol', b=1)
    try:
        cp2(os.path.join(seed_dirs[cur_seed], 'md.edr'), '{}.edr'.format(new_name))
    except:
        print('Error when tried to copy energy file. Maybe you do not produce them ? Then comment this line.')
    os.remove(os.path.join(seed_dirs[cur_seed], 'md.xtc'))


def make_a_step2(work_dir, cur_seed, seed_dirs, top_file, ndx_file, seed_digest_filename, old_name_digest, past_dir, hostname, ncores) -> NoReturn:
    """

    :param work_dir: path to the directory where seed dirs reside
    :param cur_seed: current seed value used for MD production
    :param seed_dirs: dict which contains physical path to the directory where simulation with particular seed is performed
    :param top_file: .top - topology of the current conformation
    :param ndx_file: .ndx - index of the protein atoms of the current conformation
    :param seed_digest_filename: digest for a current MD simulation, used to store files in the past
    :param old_name_digest: digest for a prior MD simulation
    :param past_dir: path to the directory with prior computations
    :param hostname: hostname to use for MD simulation
    :param ncores: number of cores to use for this task
    """
    # global extra_past
    old_name = os.path.join(past_dir, old_name_digest)
    if not os.path.exists(old_name + '.gro'):
        # old_name = os.path.join(extra_past, old_name_digest)
        # if not os.path.exists(old_name + '.gro'):
        raise Exception("make_a_step2: did not find {} in {}".format(old_name_digest, past_dir))
    gmx_grompp(work_dir, cur_seed, top_file, old_name)
    new_name = os.path.join(past_dir, seed_digest_filename)
    gmx_mdrun_mpi(work_dir, cur_seed, new_name + '.gro', hostname, ncores)
    gmx_trjconv(f=os.path.join(seed_dirs[cur_seed], 'md.xtc'), o='{}.xtc'.format(new_name),
                n=ndx_file, s=os.path.join(seed_dirs[cur_seed], 'md.tpr'), pbc='mol', b=1)
    try:
        cp2(os.path.join(seed_dirs[cur_seed], 'md.edr'), '{}.edr'.format(new_name))
    except:
        print('Error when tried to copy energy file. Maybe you do not produce them ? Then comment this line.')
    os.remove(os.path.join(seed_dirs[cur_seed], 'md.xtc'))


def make_a_step3(work_dir: str, cur_seed, seed_dirs, top_file, ndx_file, seed_digest_filename, old_name_digest, past_dir, ncores, ntomp=1) -> NoReturn:
    """

    :param work_dir:  path to the directory where seed dirs reside
    :param cur_seed: current seed value used for MD production
    :param seed_dirs: dict which contains physical path to the directory where simulation with particular seed is performed
    :param top_file: .top - topology of the current conformation
    :param ndx_file: .ndx - index of the protein atoms of the current conformation
    :param seed_digest_filename: digest for a current MD simulation, used to store files in the past
    :param old_name_digest: digest for a prior MD simulation
    :param past_dir: path to the directory with prior computations
    :param ncores: number of cores to use for this task
    :param ntomp: number of OMP threads to use during the simulation
    """
    # global extra_past
    old_name = os.path.join(past_dir, old_name_digest)
    if not os.path.exists(old_name +'.gro'):
        # old_name = os.path.join(extra_past, old_name_digest)
        # if not os.path.exists(old_name + '.gro'):
        raise Exception("make_a_step3: did not find {} in {}".format(old_name_digest, past_dir))
    gmx_grompp(work_dir, cur_seed, top_file, old_name)
    new_name = os.path.join(past_dir, seed_digest_filename)
    # gmx_mdrun_mpi(work_dir, cur_seed, new_name + '.gro', hostname, ncores)
    gmx_mdrun_mpi_with_sched(work_dir, cur_seed, new_name + '.gro', ncores, ntomp)
    gmx_trjconv(f=os.path.join(seed_dirs[cur_seed], 'md.xtc'), o='{}.xtc'.format(new_name),
                n=ndx_file, s=os.path.join(seed_dirs[cur_seed], 'md.tpr'), pbc='mol', b=1)
    try:
        cp2(os.path.join(seed_dirs[cur_seed], 'md.edr'), '{}.edr'.format(new_name))
    except:
        print('Error when tried to copy energy file. Maybe you do not produce them ? Then comment this line.')
    os.remove(os.path.join(seed_dirs[cur_seed], 'md.xtc'))


def get_seed_dirs(work_dir: str, list_with_cur_seeds: list, simulation_temp: int, sd: dict = None):
    """
    Create directories with unique names for simulation with specified seeds and puts .mdp, config files for the MD simulation.
    :param work_dir: path to work directory, where all seed directories reside
    :param list_with_cur_seeds: list of seed currently used
    :param simulation_temp: simulation temperature used to generate proper .mdp file
    :param sd: Not used anymore, but left for sime time as deprecated. sd - previous seed deers
    :return: dictionary with seed dir paths
    :rtype dict
    """
    if not sd:
        sd = dict()
    for seed in list_with_cur_seeds:
        seed_dir = os.path.join(work_dir, str(seed))
        sd[seed] = seed_dir
        if not os.path.exists(seed_dir):
            os.makedirs(seed_dir)
        with open(os.path.join(sd[seed], 'md.mdp'), 'w') as f:
            f.write(get_mdp(seed, simulation_temp))
    return sd


def rm_seed_dirs(seed_dirs) -> NoReturn:
    """
    Removes seed directory and all it's content
    :param seed_dirs: dict which contains physical path to the directory where simulation with particular seed is performed
    """
    for seed_dir in seed_dirs.values():
        if os.path.exists(seed_dir):
            shutil.rmtree(seed_dir, ignore_errors=True)


def get_new_seeds(old_seeds: list, seed_num: int = 4) -> list:
    """
    Returns next seed sequence.
    :param old_seeds: list of previous seeds
    :param seed_num: number of unique seeds in the current run
    :return: list of new seeds
    :rtype list
    """
    max_seeds = 64000  # change this if you want more exploration
    if min(old_seeds) + seed_num > max_seeds:
        return None
    return [seed + seed_num for seed in old_seeds]


def trjcat_many(hashed_names, past_dir, out_name) -> NoReturn:
    """
    Concatenates many trajectories into one file
    :param hashed_names: .xtc filenames to concatenate
    :param past_dir: path to the directory with prior computations
    :param out_name: single output filename
    """
    wave = 100
    tot_chunks = int((len(hashed_names) + 1) / wave)
    print('wave={}, tot_chunks={}'.format(wave, tot_chunks))
    gmx_trjcat(f=[os.path.join(past_dir, hashed_name) + '.xtc' for hashed_name in hashed_names[:wave]],
               o='./combinded_traj.xtc', n='./prot_dir/prot.ndx', cat=True, vel=False, sort=False, overwrite=True)
    for i in range(wave, len(hashed_names), wave):
        os.rename('./combinded_traj.xtc', './combinded_traj_prev.xtc')
        gmx_trjcat(f=[" ./combinded_traj_prev.xtc "] + [os.path.join(past_dir, hashed_name) + '.xtc' for hashed_name in hashed_names[i:i+wave]],
                   o='./combinded_traj.xtc',
                   n='./prot_dir/prot.ndx', cat=True, vel=False, sort=False, overwrite=True)
        if int(i / wave) % 10 == 0:
            print('{}/{} ({:.1f}%)'.format(int(i / wave), tot_chunks, 100 * int(i / wave) / tot_chunks))
    if os.path.exists('./combinded_traj_prev.xtc'):
        os.remove('./combinded_traj_prev.xtc')
    os.rename('./combinded_traj.xtc', out_name)


def general_bak(fname, state) -> NoReturn:
    """
    Stores variables in the picke with the specific name
    :param fname: filename for the pickle
    :param state: variables to store
    """
    if os.path.exists(os.path.join(os.getcwd(), fname)):
        try:
            os.rename(os.path.join(os.getcwd(), fname), os.path.join(os.getcwd(), fname + '_prev'))
        except Exception as e:
            # print(e)
            os.remove(os.path.join(os.getcwd(), fname))
            os.rename(os.path.join(os.getcwd(), fname), os.path.join(os.getcwd(), fname + '_prev'))

    with open(fname, 'wb') as f:
        pickle.dump(state, f)


def general_rec(fname):
    """
    Reads pickle content from the file
    :param fname: pickle filename
    :return: state from the pickle
    """
    with open(fname, 'rb') as f:
        state = pickle.load(f)
    return state


def main_state_backup(state) -> NoReturn:
    """
    Just a wrapper around the general_bak
    :param state: (visited_queue, open_queue, main_dict)
    """
    general_bak('small.pickle', state)


def supp_state_backup(state) -> NoReturn:
    """
    Just a wrapper around the general_bak
    :param state: (tol_error, seed_list, seed_dirs, seed_change_counter, skipped_counter, cur_metric_name,
                               cur_metric, counter_since_seed_changed, guiding_metric, greed_mult,
                             best_so_far_name, best_so_far, greed_count)
    """
    general_bak('big.pickle', state)


def main_state_recover():
    """
    Just a wrapper around the general_rec
    :return: state from the pickle
    """
    return general_rec('small.pickle')


def supp_state_recover():
    """
    Just a wrapper around the general_rec
    :return: state from the pickle
    """
    return general_rec('big.pickle')
