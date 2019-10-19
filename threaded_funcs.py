import multiprocessing
import os
from shutil import copy2 as cp2

from db_proc import get_db_con


def print_async(info_form_str, tup):
    """

    :param info_form_str:
    :param tup:
    """
    print(info_form_str.format(*tup))


def threaded_print(pipe):
    """
    Prints statement provided from the pipe.
    Typically, you supply formating string and options
    :param pipe:
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


def threaded_db_input(pipe, len_seeds: int):
    """
    Runs DB operation in a separate process
    :param pipe: connection with the parent
    :param len_seeds: total number of seeds
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
        pid = multiprocessing.Process(target=stmt[0], args=(con,)+stmt[1])
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


def threaded_copy(pipe):
    """
    Recieves filenames (A, B) from the pipe and tries to copy A into B
    :param pipe: connection with the parent
    :return:
    """
    stmt = pipe.get(timeout=3600)
    while stmt is not None:
        # with COPY_LOCK:
        cp2(stmt[0], stmt[1])
        pipe.task_done()
        stmt = pipe.get(timeout=1800)


def threaded_rm(pipe):
    """
    Recieves filename from the pipe and tries to remove them
    :param pipe: connection with the parent
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
