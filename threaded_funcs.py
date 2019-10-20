"""This file contains functions executed in a separate process to reduce I/O.
While I know that there is asyncio, but I believe that kernel can handle processes much better than Python.
Additionally, you do not create context for a function during each call, but only once - during the initial call.

    :platform: linux

.. moduleauthor:: Ivan Syzonenko <is2k@mtmail.mtsu.edu>
"""
__license__ = "MIT"
__docformat__ = 'reStructuredText'


import multiprocessing as mp
import os
from shutil import copy2 as cp2
from typing import NoReturn
from db_proc import get_db_con


def print_async(info_form_str: str, tup: tuple) -> NoReturn:
    """Test function used for async printing

    Args:
        :param str info_form_str: formatting string.
        :param tuple tup: data to print.

    Returns:
    Simply prints the string.
    """
    print(info_form_str.format(*tup))


def threaded_print(pipe: mp.JoinableQueue) -> NoReturn:
    """Prints statement provided from the pipe.

    Typically, you supply formating string and options

    Args:
        :param mp.JoinableQueue pipe: source of the perforated strings and values (str, vals).

    Returns:
    Simply prints the string.
    """
    stmt = pipe.get(timeout=3600)
    while stmt is not None:
        try:
            # with PRINT_LOCK:
            #     print(stmt[0].format(*stmt[1]))
            print(stmt[0].format(*stmt[1]))
        except Exception as e:
            print(e)
        finally:
            pipe.task_done()
            stmt = pipe.get()
    print('Print thread exiting...')


def threaded_db_input(pipe: mp.JoinableQueue, len_seeds: int) -> NoReturn:
    """Runs DB operation in a separate process

    Args:
        :param pipe: connection with the parent.
        :param len_seeds: total number of seeds.

    Returns:
    Executes the queries from the queue.
    """
    con, dbname = get_db_con(len_seeds)
    stmt = pipe.get(timeout=3600)
    pid = None
    while stmt is not None:
        try:
            pid.join()
        except Exception as e:
            if pid:
                print(e)
        # try:
        # con = con = lite.connect(dbname, timeout=3000, check_same_thread=False, isolation_level=None)
        # con.commit()
        pid = mp.Process(target=stmt[0], args=(con,)+stmt[1])
        pid.start()
        # except Exception as e:
        #     print('Found exception in db input:')
        #     print(e)
        #     print('Arguments that caused exception: ')
        #     print(stmt)
        # finally:
        pipe.task_done()
        stmt = pipe.get()
    print('DB thread exiting...')
    con.close()


def threaded_copy(pipe: mp.JoinableQueue) -> NoReturn:
    """Recieves filenames (A, B) from the pipe and tries to copy A into B

    Args:
        :param pipe: connection with the parent

    Returns:
    Copies files in the background.
    """
    stmt = pipe.get(timeout=3600)
    while stmt is not None:
        # with COPY_LOCK:
        cp2(stmt[0], stmt[1])
        pipe.task_done()
        stmt = pipe.get(timeout=1800)


def threaded_rm(pipe: mp.JoinableQueue) -> NoReturn:
    """Recieves filename from the pipe and tries to remove them

    Args:
        :param pipe: connection with the parent

    Returns:
    Removes files in the background.
    """
    stmt = pipe.get(timeout=3600)
    while stmt is not None:
        # with RM_LOCK:
        try:
            os.remove(stmt)
        except Exception as e:
            print('Was not able to remove {}, Error: {}'.format(stmt, e))
        pipe.task_done()
        stmt = pipe.get(timeout=1800)
