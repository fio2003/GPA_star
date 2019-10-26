"""
This file contains GROMACS wrappers.
    :platform: linux

.. moduleauthor:: Ivan Syzonenko <is2k@mtmail.mtsu.edu>
"""
__license__ = "MIT"
__docformat__ = 'reStructuredText'

import subprocess
import multiprocessing
import os
from typing import NoReturn, Mapping, Sequence, List, Set

my_env = os.environ.copy()
my_env["GMX_MAXBACKUP"] = "-1"
my_env["GMX_NO_QUOTES"] = ""
os.environ.update(my_env)


def convert_gro_to_xtc(gro_file: str, ndx_file: str) -> str:
    """Converts .gro into .xtc format. Just a wrapper around trjconv.

    Args:
        :param str gro_file: input filename
        :param str ndx_file: index file, shows which atoms to store in .xtc

    Returns:
        :return: .xtc filename
    """
    out_filename = gro_file[0:-3] + 'xtc'
    gmx_trjconv(f=gro_file, o=out_filename, n=ndx_file)
    return out_filename


def gmx_trjconv(f: str, o: str, n: str = None, s: str = None, b: int = None, e: int = None,
                dump: int = None, fit: str = None, vel: str = None, pbc: str = None) -> NoReturn:
    """'gmx trjconv' - GROMACS tool - converts trajectory files in many ways

    Converts between various formats. In our case from .gro to .xtc or
     from .gro to .gro with specific index file to filter protein only or it's specific parts.

    Args:
        :param str f: Input trajectory: xtc trr cpt gro g96 pdb tng
        :param str o: Output trajectory: xtc trr gro g96 pdb tng
        :param str n: Index file
        :param str s: Structure+mass(db): tpr gro g96 pdb brk ent
        :param int b: Time of first frame to read from trajectory (default unit ps)
        :param int e: Time of last frame to read from trajectory (default unit ps)
        :param int dump: Dump frame nearest specified time (ps)
        :param str fit: Fit molecule to ref structure in the structure
        file: none, rot+trans, rotxy+transxy, translation, transxy, progressive
        :param str vel: Read and write velocities if possible
        :param str pbc: PBC treatment (see help text for full description):
        none, mol, res, atom, nojump, cluster, whole

    Returns:
    Generates one output file passed with -o parameter.
    """
    if not (f and o):
        raise Exception('Missing in/out arguments.')
    command_trjconv = 'gmx trjconv -f {:s} -o {:s} '.format(f, o)
    if n:
        command_trjconv += '-n {} '.format(n)
    if s:
        command_trjconv += '-s {} '.format(s)
    if b:
        command_trjconv += '-b {} '.format(b)
    if e:
        command_trjconv += '-e {} '.format(e)
    if dump:
        command_trjconv += '-dump {} '.format(dump)
    # if vel:
    #     command_trjconv += '-vel '
    # else:
    #     command_trjconv += '-novel '
    if fit:
        if fit not in ['none', 'rot+trans', 'rotxy+transxy', 'translation', 'transxy', 'progressive']:
            raise Exception('Wrong fit parameter in gmx_trjconv.')
        command_trjconv += '-fit {} '.format(fit)
    if pbc:
        if pbc not in ['none', 'mol', 'res', 'atom', 'nojump', 'cluster', 'whole']:
            raise Exception('Wrong pbc parameter in gmx_trjconv.')
        command_trjconv += '-pbc {} '.format(pbc)

    # command_trjconv = os.path.expandvars(command_trjconv)
    # print(command_trjconv)
    proc_obj = subprocess.Popen(command_trjconv, stdout=-1, shell=True, cwd='.', stderr=-1, env=my_env)
    output, error = proc_obj.communicate()
    error = error.decode("utf-8")
    if 'error' in error.lower():
        print(error)
    # print(output.decode("utf-8"))
    # print(error)


def gmx_trjcat(f: str, o: str, n: str, cat: bool = True, vel: bool = False, sort: bool = False, overwrite: bool = True) -> NoReturn:
    """'gmx trjcat' - GROMACS tool - concatenates several input trajectory files in sorted order

    Outputs one .xtc file that contains all frames (99% frames are NOT sorted, since trajectories have the same time)

    Args:
        :param str f: Input trajectory: xtc trr cpt gro g96 pdb tng
        :param str o: Output trajectory: xtc trr gro g96 pdb tng
        :param str n: Index file
        :param bool cat: Do not discard double time frames
        :param bool vel: Read and write velocities if possible
        :param bool sort: Sort trajectory files (not frames)
        :param bool overwrite: Overwrite overlapping frames during appending

    Returns:
    Generates one output file passed with -o parameter.
    """
    command_trjcat = 'gmx trjcat -keeplast '
    if not (f and o):
        raise Exception('Missing in/out arguments.')
    command_trjcat += '-o {:s} '.format(o)
    if isinstance(f, list):
        command_trjcat += '-f ' + ' '.join(f) + ' '
    else:
        command_trjcat += '-f {:s} '.format(f)
    if n:
        command_trjcat += '-n {} '.format(n)
    if cat:
        command_trjcat += '-cat '
    else:
        command_trjcat += '-nocat '
    # if vel:
    #     command_trjcat += '-vel '
    # else:
    #     command_trjcat += '-novel '
    if sort:
        command_trjcat += '-sort '
    else:
        command_trjcat += '-nosort '
    if overwrite:
        command_trjcat += '-overwrite '

    command_trjcat = os.path.expandvars(command_trjcat)
    proc_obj = subprocess.Popen(command_trjcat, stdout=-1, shell=True, cwd='.', stderr=-1, env=my_env)
    output, error = proc_obj.communicate()
    error = error.decode("utf-8")
    if 'error' in error.lower():
        print(error)


def gmx_eneconv(f: str, o: str) -> NoReturn:
    """'gmx eneconv' - GROMACS tool - Concatenates several energy files in sorted order

    Stores converted energy files. Not used by main algorithm, but during the postprocessing.

    Args:
        :param str f: Input trajectory: xtc trr cpt gro g96 pdb tng
        :param str o: Output trajectory: xtc trr gro g96 pdb tng

    Returns:
    Generates one output energy file passed with -o parameter.
    """
    command_eneconv = 'gmx eneconv '
    if not (f and o):
        raise Exception('Missing in/out arguments.')
    command_eneconv += '-o {:s} '.format(o)
    if isinstance(f, list):
        command_eneconv += '-f ' + ' '.join(f) + ' -nosort -settime '
        #command_eneconv += '-f ' + ' '.join(f) + ' -settime '
        # command_eneconv = 'echo -e "{}" | '.format('\n'.join([str(i) for i in range(0, len(f) * 20, 20)])) + command_eneconv
        command_eneconv = 'echo -e "{}" | '.format('\n'.join(['c']*(len(f)+1))) + command_eneconv
    else:
        command_eneconv += '-f {:s} '.format(f)

    command_eneconv = os.path.expandvars(command_eneconv)
    proc_obj = subprocess.Popen(command_eneconv, stdout=-1, shell=True, cwd='.', stderr=-1, env=my_env)
    output, error = proc_obj.communicate()
    error = error.decode("utf-8")
    if 'error' in error.lower():
        print(error)


def gmx_energy(f: str, o: str, w: bool = None, w_prog: str = None, fee: bool = True, fetemp: float = 300) -> NoReturn:
    """'gmx trjconv' - GROMACS tool - extracts energy components from an energy file

    Args:
        :param str f: .edr Energy file
        :param str o: energy.xvg - xvgr/xmgr file
        :param str w: View output .xvg, .xpm, .eps and .pdb files
        :param str w_prog: viewing programm
        :param bool fee: Do a free energy estimate
        :param float fetemp: Reference temperature for free energy calculation

    Returns:
    Generates one output .xvg file passed with -o parameter.
    """
    command_energy = 'gmx energy '
    command_energy += ' -f ' + f
    command_energy += ' -o ' + o
    if w:
        command_energy += '-w {} {} '.format(w, w_prog)
    if fee:
        command_energy += ' -fee '
    if fetemp:
        command_energy += ' -fetemp {}'.format(fetemp)
    command_energy = 'echo -e "10" | ' + command_energy
    command_energy = os.path.expandvars(command_energy)
    proc_obj = subprocess.Popen(command_energy, stdout=-1, shell=True, cwd='.', stderr=-1, env=my_env)
    output, error = proc_obj.communicate()
    error = error.decode("utf-8")
    if 'error' in error.lower():
        print(error)


def gmx_mdrun(work_dir: str, seed: int, new_name: str, ncores: int = multiprocessing.cpu_count(), thread_type: str = 'nt') -> NoReturn:
    """gmx mdrun - localhost version.

    Args:
        :param str work_dir: path to work directory, where all seed directories reside
        :param int seed: seed value used in the MD simulation
        :param str new_name: output name for a final state
        :param int ncores: number of cores to use in the current simulation
        :param str thread_type: thread type: MPI ? OMP ? TMPI ?

    Returns:
    Starts a shell in a separate process and runs mdrun there.
    """
    if thread_type not in ['nt', 'ntomp']:  # 'ntmpi' is prohibited when gromacs compiled without mpi support
        raise Exception('Wrong thread type passed in gmx_mdrun')
    ncores = ncores if ncores > 0 else 1

    command_run_md = "gmx mdrun -deffnm md -{} {} -c {} -reprod".format(thread_type, ncores, new_name)
    # command_run_md = "gmx mdrun -deffnm md -{} {} -c {} -pin on -reprod".format(thread_type, ncores, new_name)
    proc_obj = subprocess.Popen(command_run_md, stdout=-1, shell=True, cwd='{}/{}/'.format(work_dir, seed), stderr=-1, env=my_env)
    output, error = proc_obj.communicate()
    error = error.decode("utf-8")
    output = output.decode("utf-8")
    # with open(str(os.getpid())+'_err.log', 'a') as log_out:
    #     log_out.write(error)
    # with open(str(os.getpid())+'_out.log', 'a') as log_out:
    #     log_out.write(output.decode("utf-8"))

    if 'error' in error.lower():
        print(error)


def gmx_mdrun_mpi(work_dir: str, seed: int, new_name: str, hostnames: list, ncores: int = None, thread_type: str = 'ntomp') -> NoReturn:
    """gmx mdrun - MPI version

    Args:
        :param str work_dir: path to work directory, where all seed directories reside
        :param int seed: seed value used in the MD simulation
        :param str new_name: output name for a final state
        :param list hostnames: must be a list
        :param int ncores: number of cores to use in the current simulation
        :param str thread_type: type of the thread, OMP ? MPI ?

    Returns:
    Starts a shell in a separate process and runs mdrun there.
    This version uses MPI to run on a separate host
    """
    if thread_type not in ['ntmpi', 'ntomp']:  # 'nt' is prohibited when gromacs compiled with mpi support
        raise Exception('Wrong thread type passed in gmx_mdrun')
    one_host_only_mpi = True
    if one_host_only_mpi:
            command_run_md = "mpirun -host {} -np 1 mdrun -deffnm md -c {} -nt 32 -ntomp 2 -pin on -reprod \
                             ".format(','.join(hostnames), new_name, int(ncores))
    else:
        if ncores:
            command_run_md = "mpirun -host {0} -np {1} mdrun -deffnm md -c {2} -ntomp 2 -nt {1} -pin on -reprod \
                             ".format(','.join(hostnames), min(1, int(ncores)), new_name)
            # command_run_md = "mpirun -host {} -np {} mdrun_mpi -deffnm md -c {} -ntomp 2 -pin on -reprod \
            #                  ".format(','.join(hostnames), min(1, int(ncores)//2), new_name)
        else:
            command_run_md = "mpirun -hosts {} gmx mdrun -deffnm md -c {} -ntomp 2 -pin on -reprod".format(','.join(hostnames), new_name)
    proc_obj = subprocess.Popen(command_run_md, stdout=-1, shell=True, cwd='{}/{}/'.format(work_dir, seed), stderr=-1, env=my_env)
    output, error = proc_obj.communicate()
    error = error.decode("utf-8")
    output = output.decode("utf-8")
    # with open(str(os.getpid())+'_err.log', 'a') as log_out:
    #     log_out.write(error)
    # with open(str(os.getpid())+'_out.log', 'a') as log_out:
    #     log_out.write(output.decode("utf-8"))

    if 'error' in error.lower():
        print(error)


def gmx_mdrun_mpi_with_sched(work_dir: str, seed: int, new_name: str, ncores: list = None, ntomp: int = 1) -> NoReturn:
    """gmx mdrun - MPI version with scheduler

    Args:
        :param str work_dir: path to work directory, where all seed directories reside
        :param int seed: seed value used in the MD simulation
        :param str new_name: output name for a final state
        :param list ncores: number of cores to use in the current simulation
        :param int ntomp: number of OMP threads

    Returns:
    Starts a shell in a separate process and runs mdrun there.
    This version uses MPI but does not specify the host, it should be done through the scheduler.
    Do not use this version if you know the exact host names - then you have more control and potentially less overhead.
    """
    if ncores % ntomp != 0 or (ntomp > ncores):
        raise Exception('Not possible to divide OMP threads evenly among the specified number of cores.\nCores: {}\tOMP threads: {}\n'.format(ncores, ntomp))

    if ntomp == ncores:
        command_run_md = "mpirun -np {0} mdrun -deffnm md -c {1} -pin on -reprod".format(ncores, new_name)
    else:
        command_run_md = "mpirun -np {0} mdrun -deffnm md -c {1} -ntomp {2} -pin on -reprod".format(ncores, new_name, ntomp)

    proc_obj = subprocess.Popen(command_run_md, stdout=-1, shell=True, cwd='{}/{}/'.format(work_dir, seed), stderr=-1, env=my_env)
    output, error = proc_obj.communicate()
    error = error.decode("utf-8")
    output = output.decode("utf-8")
    # with open(str(os.getpid())+'_err.log', 'a') as log_out:
    #     log_out.write(error)
    # with open(str(os.getpid())+'_out.log', 'a') as log_out:
    #     log_out.write(output.decode("utf-8"))

    if 'error' in error.lower():
        print(error)


def gmx_grompp(work_dir: str, seed: int, top_file: str, prev_name: str) -> NoReturn:
    """gmx grompp (the gromacs preprocessor) reads a molecular topology file, checks the validity of the file,
     expands the topology from a molecular description to an atomic description.

    Args::
    
        :param str work_dir: path to work directory, where all seed directories reside
        :param int seed: seed value used in the MD simulation
        :param str top_file: .top - topology of the conformation
        :param str prev_name: previous simulation digest. Used as starting point.

    Returns
    
    Creates .tpr - binary config file.
    """
    command_prep_run = "gmx grompp -f md.mdp -c {}.gro -p {} -o md.tpr".format(prev_name, top_file)
    proc_obj = subprocess.Popen(command_prep_run, stdout=-1, shell=True, cwd=os.path.join(work_dir, str(seed)), stderr=-1, env=my_env)
    output, error = proc_obj.communicate()
    error = error.decode("utf-8")
    # with open(str(os.getpid())+'_err.log', 'a') as log_out:
    #     log_out.write(error)
    # with open(str(os.getpid())+'_out.log', 'a') as log_out:
    #     log_out.write(output.decode("utf-8"))

    if 'error' in error.lower():
        print(error)
