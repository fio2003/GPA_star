#!/usr/bin/env python3.6

"""
This file contains various wrappers and functions that ease the code digestion and programming in general.
    .. module:: main
    :platform: linux

.. moduleauthor:: Ivan Syzonenko <is2k@mtmail.mtsu.edu>
"""
__license__ = "MIT"
__docformat__ = 'reStructuredText'

import multiprocessing
import os
from GMDA_main import GMDA_main
from threaded_funcs import threaded_db_input, threaded_print  # ,threaded_copy, threaded_rm
# from helper_funcs import get_previous_runs_info


def main():
    """This function is basically a launcher

    Parallel threads did not result in a much better performance and was masked for better times.
    However, if you decide to implement C++ parallel I/O - it should help.
    """
    # Compilation steps:
    # compile latest gcc
    # compile gromacs with shared libs and static libs, without mpi; install
    # compile mdsctk
    # OPTIONAL: compile gromacs with mpi/openmp if needed.
    tot_seeds = 4
    # get_db_con(tot_seeds=4)

    past_dir = os.path.join(os.getcwd(), 'past/')
    #
    # PRINT_LOCK = Lock()
    # COPY_LOCK = Lock()
    # RM_LOCK = Lock()

    # print_queue = queue.Queue()
    # printing_thread = Thread(target=threaded_print, args=(print_queue,))
    # printing_thread.start()

    # db_input_queue = queue.Queue()
    # db_input_thread = Thread(target=threaded_db_input, args=(db_input_queue, tot_seeds,))
    # db_input_thread.start()
    # # db_input_queue.put(None)
    #
    # copy_queue = queue.Queue()
    # copy_thread = Thread(target=threaded_copy, args=(copy_queue,))
    # copy_thread.start()
    #
    # rm_queue = queue.Queue()
    # rm_thread = Thread(target=threaded_rm, args=(rm_queue, RM_LOCK,))
    # rm_thread.start()

    # prev_runs_files = get_previous_runs_info(past_dir)

    # print_queue = multiprocessing.JoinableQueue(102400)
    # printing_thread = multiprocessing.Process(target=threaded_print, args=(print_queue,))
    # printing_thread.start()
    print_queue = None

    db_input_queue = multiprocessing.JoinableQueue(102400)
    db_input_thread = multiprocessing.Process(target=threaded_db_input, args=(db_input_queue, tot_seeds,))
    db_input_thread.start()

    # no need in the next queues. Maybe helpful if working with /dev/shm
    # copy_queue = None
    # copy_queue = multiprocessing.Queue()
    # copy_thread = multiprocessing.Process(target=threaded_copy, args=(copy_queue,))
    # copy_thread.start()

    # rm_queue = None
    # rm_queue = multiprocessing.JoinableQueue(3)
    # rm_thread = multiprocessing.Process(target=threaded_rm, args=(rm_queue,))
    # rm_thread.start()

    GMDA_main(past_dir, print_queue, db_input_queue, tot_seeds)
    # GMDA_main(prev_runs_files, past_dir, print_queue, db_input_queue, copy_queue, rm_queue, tot_seeds)

    print_queue.put_nowait(None)
    db_input_queue.put_nowait(None)
    printing_thread.join()
    db_input_thread.join()
    print('The last line of the program.')
    # rm_queue.put_nowait(None)
    # print_queue.join()
    # db_input_queue.join()
    # rm_queue.join()


if __name__ == "__main__":
    main()
