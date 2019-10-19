#!/usr/bin/env python3

import sqlite3 as lite
import os
import sys
from gmx_wrappers import gmx_trjcat
import sqlite3 as lite
import os
# import matplotlib.pyplot as plt
# import scipy
# from scipy.optimize import curve_fit
# import numpy as np
# from matplotlib.ticker import NullFormatter  # useful for `logit` scale
# from matplotlib import gridspec
# from PIL import Image
# from matplotlib import figure
# from matplotlib.figure import figaspect
from gmx_wrappers import gmx_eneconv, gmx_energy
from shutil import copy2
import multiprocessing as mp


def main():
    db_to_connect = 'results_opls_trp_300_fixed'
    # if len(sys.argv) < 2:
    #     raise Exception('Not enough arguments')
    # db_to_connect = sys.argv[1]
    # try:
    #     os.mkdir('best_past')
    # except:
    #     pass
    for metr in ['rmsd', 'angl', 'andh', 'and', 'xor']:
        build_best_traj(metr, db_to_connect)
    # pool = mp.Pool(len(['rmsd', 'angl', 'andh', 'and', 'xor']))  # we are IO bound in graphs, no need to use exact number of CPUs mp.cpu_count()
    # results1 = pool.starmap_async(build_best_traj, [(metr, db_to_connect) for metr in ['rmsd', 'angl', 'andh', 'and', 'xor']])
    # results1.get()
    # pool.close()



def build_best_traj(metr_name, db_to_connect):

    # db_to_connect = 'results_opls_trp_300_2_fixed'

    past_dir = './past'
    if not os.path.exists(db_to_connect + '.sqlite3'):
        raise Exception('DB not found')

    con = lite.connect(db_to_connect + '.sqlite3', check_same_thread=False, isolation_level=None)
    cur = con.cursor()

    qry = "select a.name, a.hashed_name, a.{0}_goal_dist from main_storage a \
           where a.{0}_goal_dist= ( select min(b.{0}_goal_dist) from main_storage b)".format(metr_name)
    result = cur.execute(qry)
    all_res = result.fetchone()
    print('The closest frame to goal has {} {} and name:\n{}'.format(metr_name, all_res[2], all_res[1]))
    name = all_res[0]
    spname = name.split('_')
    all_prev_names = ['\'{}\''.format('_'.join(spname[:i])) for i in range(1, len(spname)+1)]
    long_line = ", ".join(all_prev_names)

    qry = "select name, hashed_name from main_storage where name in ({})".format(long_line)
    result = cur.execute(qry)
    all_res = result.fetchall()
    con.close()

    names, hashed_names = zip(*all_res)

    # for file in [os.path.join(past_dir, hashed_name) for hashed_name in hashed_names]:
    #     copy2('{}.xtc'.format(file), './best_past/')
    #     try:
    #         copy2('{}.edr'.format(file), './best_past/')
    #     except:
    #         print('Failed to copy {}; Normal for the first frame.'.format(file))

    wave = 100
    tot_chunks = int((len(hashed_names) + 1) / wave)
    print('Computing best trajectory for {}'.format(metr_name))
    print('wave={}, tot_chunks={}'.format(wave, tot_chunks))
    if os.path.exists('./{}_combined_traj.xtc'.format(metr_name)):
        os.remove('./{}_combined_traj.xtc'.format(metr_name))
    if os.path.exists('./{}_combined_traj_prev.xtc'.format(metr_name)):
        os.remove('./{}_combined_traj_prev.xtc'.format(metr_name))

    gmx_trjcat(f=[os.path.join(past_dir, hashed_name) + '.xtc' for hashed_name in hashed_names[:wave]],
               o='./{}_combined_traj.xtc'.format(metr_name), n='./prot_dir/prot_unfolded.ndx', cat=True, vel=False, sort=False, overwrite=True)
    for i in range(wave, len(hashed_names), wave):
        os.rename('./{}_combined_traj.xtc'.format(metr_name), './{}_combined_traj_prev.xtc'.format(metr_name))
        gmx_trjcat(f=[" ./{}_combined_traj_prev.xtc ".format(metr_name)] + [os.path.join(past_dir, hashed_name) + '.xtc' for hashed_name in hashed_names[i:i+wave]],
                   o='./{}_combined_traj.xtc'.format(metr_name), n='./prot_dir/prot_unfolded.ndx', cat=True, vel=False, sort=False, overwrite=True)
        if int(i / wave) % 10 == 0:
            print('{}/{} ({:.1f}%)'.format(int(i / wave), tot_chunks, 100 * int(i / wave) / tot_chunks))

    if os.path.exists('./{}_combined_traj.xtc'.format(metr_name)):
        os.rename('./{}_combined_traj.xtc'.format(metr_name), './{}_{}_traj_best.xtc'.format(metr_name, db_to_connect))
    if os.path.exists('./{}_combined_traj_prev.xtc'.format(metr_name)):
        os.remove('./{}_combined_traj_prev.xtc'.format(metr_name))
    print('Done with best for {}: {}'.format(metr_name, db_to_connect))


    # ###### ENERGIES
    if os.path.exists('./{}_combined_energy.edr'.format(metr_name)):
        os.remove('./{}_combined_energy.edr'.format(metr_name))
    if os.path.exists('./{}_combined_energy_prev.edr'.format(metr_name)):
        os.remove('./{}_combined_energy_prev.edr'.format(metr_name))
    hashed_names = hashed_names[1:]
    tot_chunks = int((len(hashed_names) + 1) / wave)
    print('Computing energy for best trajectory for {}'.format(metr_name))
    print('wave={}, tot_chunks={}'.format(wave, tot_chunks))
    gmx_eneconv(f=[os.path.join("./past", hashed_name) + '.edr' for hashed_name in hashed_names[:wave]], o='./{}_combined_energy.edr'.format(metr_name))
    for i in range(wave, len(hashed_names), wave):
        os.rename('./{}_combined_energy.edr'.format(metr_name), './{}_combined_energy_prev.edr'.format(metr_name))
        gmx_eneconv(f=["./{}_combined_energy_prev.edr".format(metr_name)] + [os.path.join("./past", hashed_name + '.edr') for hashed_name in hashed_names[i:i + wave if i + wave < len(hashed_names) else -1]],
                    o='./{}_combined_energy.edr'.format(metr_name))
        if int(i / wave) % 10 == 0:
            print('{}/{} ({:.1f}%)'.format(int(i / wave), tot_chunks, 100 * int(i / wave) / tot_chunks))

    os.rename('./{}_combined_energy.edr'.format(metr_name), './{}_combined_energy_best.edr'.format(metr_name))


if __name__ == '__main__':
    main()





def main_energy():
    past_dir = './past'
    db_to_connect = 'results_12'
    polynomial = False
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 16,
            }
    if not os.path.exists(db_to_connect + '.sqlite3'):
        raise Exception('DB not found')

    con = lite.connect(db_to_connect + '.sqlite3', check_same_thread=False, isolation_level=None)
    cur = con.cursor()

    qry = "select a.name, a.hashed_name from main_storage a  where a.goal_dist= ( select min(b.goal_dist) from main_storage b)"
    result = cur.execute(qry)
    all_res = result.fetchone()
    name = all_res[0]
    spname = name.split('_')
    all_prev_names = ['\'{}\''.format('_'.join(spname[:i])) for i in range(1, len(spname))]
    long_line = ", ".join(all_prev_names)

    qry = "select name, hashed_name from main_storage where name in ({})".format(long_line)
    result = cur.execute(qry)
    _ = result.fetchone()
    all_res = result.fetchall()
    names, hashed_names = zip(*all_res)
    wave = 100
    tot_chunks = int((len(hashed_names) + 1) / wave)
    print('wave={}, tot_chunks={}'.format(wave, tot_chunks))
    gmx_eneconv(f=[os.path.join("./past", hashed_name) + '.edr' for hashed_name in hashed_names[:wave]], o='./combined_energy.edr')
    for i in range(wave, len(hashed_names) + 1 - wave, wave):
        os.rename('./combined_energy.edr', './combined_energy_prev.edr')
        gmx_eneconv(f=["./combined_energy_prev.edr"] + [os.path.join("./past", hashed_name + '.edr') for hashed_name in hashed_names[i:i + wave if i + wave < len(hashed_names) else -1]],
                    o='./combined_energy.edr')
        if int(i / wave) % 10 == 0:
            print('{}/{} ({:.1f}%)'.format(int(i / wave), tot_chunks, 100 * int(i / wave) / tot_chunks))

    os.rename('./combined_energy.edr', './combined_energy_best.edr')
    print('Done with best')



    qry = "select a.name, a.hashed_name from main_storage a "
    result = cur.execute(qry)
    _ = result.fetchone()
    all_res = result.fetchall()
    names, hashed_names = zip(*all_res)

    # gmx_eneconv(f=[os.path.join(past_dir, hash_name+'.edr') for hash_name in hashed_names], o='./combined_energy.edr')

    wave = 100
    tot_chunks = int((len(hashed_names)+1)/wave)
    print('wave={}, tot_chunks={}'.format(wave, tot_chunks))
    gmx_eneconv(f=[os.path.join("./past", hashed_name)+'.edr' for hashed_name in hashed_names[:wave]], o='./combined_energy.edr')
    for i in range(wave, len(hashed_names)+1-wave, wave):
        os.rename('./combined_energy.edr', './combined_energy_prev.edr')
        gmx_eneconv(f=["./combined_energy_prev.edr"] +[os.path.join("./past", hashed_name + '.edr') for hashed_name in hashed_names[i:i+wave if i+wave < len(hashed_names) else -1]], o='./combined_energy.edr')
        if int(i/wave) % 10 == 0:
            print('{}/{} ({:.1f}%)'.format(int(i/wave), tot_chunks, 100*int(i/wave)/tot_chunks))

    os.rename('./combined_energy.edr', './combined_energy_all_main.edr')
    print('Done with all main')


    qry = "select a.name, a.hashed_name from main_storage a join log b on a.id=b.id where b.dst='VIZ' order by b.timestamp"
    result = cur.execute(qry)
    _ = result.fetchone()
    all_res = result.fetchall()
    names, hashed_names = zip(*all_res)

    wave = 100
    tot_chunks = int((len(hashed_names)+1)/wave)
    print('wave={}, tot_chunks={}'.format(wave, tot_chunks))
    gmx_eneconv(f=[os.path.join("./past", hashed_name)+'.edr' for hashed_name in hashed_names[:wave]], o='./combined_energy.edr')
    for i in range(wave, len(hashed_names)+1-wave, wave):
        os.rename('./combined_energy.edr', './combined_energy_prev.edr')
        gmx_eneconv(f=["./combined_energy_prev.edr"] +[os.path.join("./past", hashed_name + '.edr') for hashed_name in hashed_names[i:i+wave if i+wave < len(hashed_names) else -1]], o='./combined_energy.edr')
        if int(i/wave) % 10 == 0:
            print('{}/{} ({:.1f}%)'.format(int(i/wave), tot_chunks, 100*int(i/wave)/tot_chunks))

    os.rename('./combined_energy.edr', './combined_energy_all_viz.edr')
    print('Done with viz')


    # gmx_energy('./combined_energy.edr', './combined_energy.xvg', fee=True, fetemp=300)


