"""
This file contains DB related functions.
.. module:: GMDA_main
    :platform: linux

.. moduleauthor:: Ivan Syzonenko <is2k@mtmail.mtsu.edu>
"""
__license__ = "MIT"
__docformat__ = 'reStructuredText'

import os
import sqlite3 as lite
import numpy as np
lite.register_adapter(np.int64, lambda val: int(val))
lite.register_adapter(np.int32, lambda val: int(val))
lite.register_adapter(np.float, lambda val: float(val))
lite.register_adapter(np.float32, lambda val: float(val))
# import numpy as np
from typing import NoReturn, Mapping, Sequence, List, Set


def get_db_con(tot_seeds: int = 4) -> tuple:
    """Creates the database with structure that fits exact number of seeds.

    Filename for DB is generated as next number after the highest consequent found.
    If there is results_0.sqlite3, then next will be results_1.sqlite3 if it did not exist.

    Args:
        :param int tot_seeds: number of seeds used in the current run
        :type tot_seeds: int

    Returns:
        :return: database connection and name

    Connection to the new database and it's name.
    """
    counter = 0
    # db_path = '/dev/shm/GMDApy'
    db_path = os.getcwd()
    db_name = 'results_{}.sqlite3'.format(counter)
    full_path = os.path.join(db_path, 'results_{}.sqlite3'.format(counter))
    while os.path.exists(full_path):
        counter += 1
        full_path = os.path.join(db_path, 'results_{}.sqlite3'.format(counter))

    con = lite.connect(full_path, check_same_thread=False, isolation_level=None)

    cur = con.cursor()
    cur.execute("""CREATE TABLE main_storage (
        id               INTEGER   PRIMARY KEY AUTOINCREMENT,

        bbrmsd_goal_dist   FLOAT    NOT NULL,
        bbrmsd_prev_dist   FLOAT    NOT NULL,
        bbrmsd_tot_dist    FLOAT    NOT NULL,
        
        aarmsd_goal_dist   FLOAT    NOT NULL,
        aarmsd_prev_dist   FLOAT    NOT NULL,
        aarmsd_tot_dist    FLOAT    NOT NULL,

        angl_goal_dist   FLOAT    NOT NULL,
        angl_prev_dist   FLOAT    NOT NULL,
        angl_tot_dist    FLOAT    NOT NULL,

        andh_goal_dist   INTEGER   NOT NULL,
        andh_prev_dist   INTEGER   NOT NULL,
        andh_tot_dist    INTEGER   NOT NULL,

        and_goal_dist    INTEGER   NOT NULL,
        and_prev_dist    INTEGER   NOT NULL,
        and_tot_dist     INTEGER   NOT NULL,

        xor_goal_dist    INTEGER   NOT NULL,
        xor_prev_dist    INTEGER   NOT NULL,
        xor_tot_dist     INTEGER   NOT NULL,

        curr_gc          INTEGER   NOT NULL,
        Timestamp        DATETIME DEFAULT (CURRENT_TIMESTAMP),
        hashed_name      CHAR (32) NOT NULL UNIQUE,
        name             TEXT
        );""")
    con.commit()
    cur.execute("""CREATE TABLE visited (
        vid        INTEGER   PRIMARY KEY AUTOINCREMENT, \
        id        REFERENCES main_storage (id),
        cur_gc    INTEGER,
        Timestamp DATETIME DEFAULT (CURRENT_TIMESTAMP)
    );""")
    con.commit()

    add_ind_q = 'CREATE INDEX viz_id_idx ON visited (id);'
    cur.execute(add_ind_q)
    con.commit()

    # id         REFERENCES main_storage (id), \
    init_query = 'CREATE TABLE log ( \
        lid        INTEGER   PRIMARY KEY AUTOINCREMENT, \
        operation  INTEGER, \
        id         INTEGER, \
        src        CHAR (8), \
        dst        CHAR(8), \
        cur_metr   CHAR(5), \
        gc         INTEGER , \
        mul        FLOAT, \
        bsfrb      FLOAT, \
        bsfr        FLOAT, \
        bsfn        FLOAT, \
        bsfh        FLOAT, \
        bsfa        FLOAT, \
        bsfx        FLOAT, \
        Timestamp  DATETIME DEFAULT (CURRENT_TIMESTAMP)'  # no this is not an error
    for i in range(tot_seeds):
        init_query += ", \
        dist_from_prev_{0} FLOAT, \
        dist_to_goal_{0}   FLOAT ".format(i+1)
    init_query += ');'

    cur.execute(init_query)
    con.commit()
    add_ind_q = 'CREATE INDEX log_id_idx ON log (id);'
    cur.execute(add_ind_q)
    con.commit()

    cur.execute('PRAGMA mmap_size=-64000')  # 32M
    cur.execute('PRAGMA journal_mode =  OFF')
    cur.execute('PRAGMA synchronous = OFF')
    cur.execute('PRAGMA temp_store = MEMORY')
    cur.execute('PRAGMA threads = 32')

    return con, db_name


def log_error(con: lite.Connection, type: str, id: int) -> NoReturn:
    """Writes an error message into the log table

    Args:
        :param con: current DB connection
        :param type: error type
        :param id: id associated with the error

    Returns:
    Adds one row in the log table.
    """
    qry = 'INSERT INTO log (id, operation, dst)  VALUES ({}, "ERROR", "{}")'.format(id, type)
    try:
        con.cursor().execute(qry)
        con.commit()
    except Exception as e:
        print(e)
        print('Error in "log_error": {}'.format(qry))


# def get_id_for_name(con, name):
#     con.commit()
#     qry = "SELECT id FROM main_storage WHERE name='{}'".format(name)
#     cur = con.cursor()
#     result = cur.execute(qry)
#     num = int(result.fetchone()[0])
#     if not isinstance(num, int):
#         raise Exception("ID was not found in main stor")
#     return num


def get_id_for_hash(con: lite.Connection, h_name: str) -> int:
    """Searches main storage for id with given hash

    Args:
        :param lite.Connection con: DB connection
        :param str h_name: hashname to use during the search

    Returns:
        :return: id or None if not found
    """
    con.commit()
    qry = "SELECT id FROM main_storage WHERE hashed_name='{}'".format(h_name)
    cur = con.cursor()
    result = cur.execute(qry)
    row = result.fetchone()
    if row is not None:
        num = int(row[0])
    else:
        num = None
    # if not isinstance(num, int):
    #     print("ID was not found in main stor")
    return num


def get_corr_vid_for_id(con: lite.Connection, max_id: int, prev_ids: list, last_gc: float) -> tuple:
    """Used for recovery procedure. Tries to find matching sequence of nodes in the visited table

    Args:
        :param lite.Connection con: DB connection
        :param int max_id: maximum value of the id (defined by previous search as the common latest id)
        :param list prev_ids: several ids that should match
        :param float last_gc: extra check, whether greed counters also match

    Returns:
        :return: last common visited id, timestamp, and id
        :rtype: tuple
    """
    qry = "SELECT vid, id, CAST(strftime('%s', Timestamp) AS INT), cur_gc FROM visited WHERE id<'{}' AND id in ({}, {}, {}) order by vid desc".format(max_id, prev_ids[0], prev_ids[1], prev_ids[2])
    cur = con.cursor()
    result = cur.execute(qry)
    rows = result.fetchall()
    i = 0
    while i+2 < len(rows):  # 3 for next version
        if rows[i][0] - rows[i+1][0] == 1 and rows[i+1][0] - rows[i+2][0] == 1:
            break
        i += 1
    if i+2 >= len(rows):
        raise Exception("Sequence of events from pickle dump not found in DB")
    last_good_vid = rows[i][0]
    last_good_ts = rows[i][2]
    last_good_id = rows[i][1]
    if last_gc != int(rows[i][3]):
        raise Exception('Everything looked good, but greed counters did not match.\n Check manually and comment this exception if you are sure that this is normal.\n')

    return last_good_vid, last_good_ts, last_good_id


def get_corr_lid_for_id(con: lite.Connection, next_id: int, vid_ts: int, last_vis_id: int) -> int:
    """
    Used for recovery procedure. Tries to find matching sequence of nodes in the log table

    Args:
        :param lite.Connection con: DB connection
        :param int next_id: next id we expect to see in the log, used for double check
        :param int vid_ts: visited timestampt
        :param int last_vis_id: last visited id

    Returns:
        :return: the latest valid log_id
    """
    qry = "SELECT lid, CAST(strftime('%s', Timestamp) AS INT) FROM log WHERE id='{}' AND src='WQ' AND dst='VIZ' order by lid".format(last_vis_id)
    cur = con.cursor()
    result = cur.execute(qry)
    rows = result.fetchall()
    if len(rows) > 1:
        # find the smallest dist between vid_ts and all ts
        dist = abs(rows[0][1] - vid_ts)
        good_lid = int(rows[0][0])
        i = 1
        while i < len(rows):
            if abs(rows[i][1] - vid_ts) <= dist:
                dist = abs(rows[i][1] - vid_ts)
                good_lid = int(rows[i][0])
            i += 1
    else:
        good_lid = int(rows[0][0])

    # so now we have good_lid which is very close, but may be not exact

    qry = "SELECT lid, operation, id, src, dst FROM log WHERE lid > {} order by lid limit 4".format(good_lid)
    result = cur.execute(qry)
    rows = result.fetchall()
    i = 0
    if (rows[i][1] == 'current' and rows[i][4] == 'WQ') or rows[i][1] == 'skip':
        good_lid += 1
        i += 1
        if rows[i][1] == 'prom_O':
            good_lid += 1
            i += 1

    if rows[i][1] == 'result' and rows[i][4] == 'VIZ' and int(rows[i][2]) == next_id:
        print("Log table ID computed perfectly.")

    return good_lid


# I am not using it
# def get_max_id_from_main(con):
#     qry = "SELECT max(id) FROM main_storage"
#     cur = con.cursor()
#     result = cur.execute(qry)
#     row = result.fetchone()
#     if row is not None:
#         num = int(row[0])
#     else:
#         num = None
#     return num


def get_all_hashed_names(con: lite.Connection) -> list:
    """Fetches all hashes from the main_storage

    Args:
        :param lite.Connection con: DB connection

    Returns:
        :return: list of all hashes in the main_storage
        :rtype: list
    """
    qry = "SELECT hashed_name FROM main_storage order by id desc"
    cur = con.cursor()
    result = cur.execute(qry)
    rows = result.fetchall()
    return rows


def insert_into_main_stor(con: lite.Connection, node_info: dict, curr_gc: int, digest_name: str, name: str) -> NoReturn:
    """Inserts main information into the DB.

    Args:
        :param lite.Connection con: DB connection
        :param dict node_info: all metric values associated with the node
        :param int curr_gc: current greedy counter
        :param str digest_name: hash name for the path, same as filenames for MD simulations
        :param str name: path from the origin separated by _

    Returns:
    Stores data in the DB in a main_storage table.
    """
    # con = lite.connect('results_8.sqlite3', timeout=300, check_same_thread=False, isolation_level=None)
    # qry = "INSERT OR IGNORE INTO main_storage(rmsd_goal_dist, rmsd_prev_dist, rmsd_tot_dist, angl_goal_dist,
    # angl_prev_dist, angl_tot_dist," \
    qry = "INSERT INTO main_storage(bbrmsd_goal_dist, bbrmsd_prev_dist, bbrmsd_tot_dist, aarmsd_goal_dist, aarmsd_prev_dist, aarmsd_tot_dist, angl_goal_dist, angl_prev_dist, angl_tot_dist," \
          "                         andh_goal_dist, andh_prev_dist, andh_tot_dist, and_goal_dist, and_prev_dist, and_tot_dist," \
          "                         xor_goal_dist, xor_prev_dist, xor_tot_dist, curr_gc, hashed_name, name) " \
          "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,   ?, ?, ?, ?, ?, ?,   ?, ?, ?, ?, ?, ?)"
    cur = con.cursor()
    try:
        cur.execute(qry, [str(elem) for elem in (node_info['BBRMSD_to_goal'], node_info['BBRMSD_from_prev'], node_info['BBRMSD_dist_total'],
                          node_info['AARMSD_to_goal'], node_info['AARMSD_from_prev'], node_info['AARMSD_dist_total'],
                          node_info['ANGL_to_goal'], node_info['ANGL_from_prev'], node_info['ANGL_dist_total'],
                          node_info['AND_H_to_goal'], node_info['AND_H_from_prev'], node_info['AND_H_dist_total'],
                          node_info['AND_to_goal'], node_info['AND_from_prev'], node_info['AND_dist_total'],
                          node_info['XOR_to_goal'], node_info['XOR_from_prev'], node_info['XOR_dist_total'],
                          curr_gc, digest_name, name)])
        con.commit()
    except Exception as e:
        nid = get_id_for_hash(con, digest_name)
        log_error(con, 'MAIN', nid)
        qry = "SELECT * FROM main_storage WHERE id=?"
        cur = con.cursor()
        result = cur.execute(qry, nid)
        row = result.fetchone()
        print('Original elment in MAIN:', row)
        qry = "SELECT * FROM log WHERE id=?"
        cur = con.cursor()
        result = cur.execute(qry, nid)
        rows = result.fetchall()
        print('Printing all I found in the log about this ID:')
        for row in rows:
            print(row)
        print('Error element message: ', e, '\nqry: ', node_info, curr_gc, digest_name, name)


def insert_into_visited(con: lite.Connection, hname: str, gc: int) -> NoReturn:
    """
    Inserts node processing event.

    Args:
        :param lite.Connection con: DB connection
        :param str hname: hashname, same as MD filenames
        :param int gc: greedy counter

    Returns:
    Stores data in the DB in a visited table.
    """
    nid = get_id_for_hash(con, hname)
    qry = 'INSERT INTO visited( id, cur_gc ) VALUES (?, ?)'
    cur = con.cursor()
    try:
        cur.execute(qry, (nid, gc))
        con.commit()
    except Exception as e:
        print(e, '\nqry: ', hname, gc)
        log_error(con, 'VIZ', nid)


def insert_into_log(con: lite.Connection, operation: str, hname: str, src: str, dst: str, bsf: list, gc: int, mul: float, prev_arr: list,
                    goal_arr: list, cur_metr_name: str) -> NoReturn:
    """Inserts various information, like new best_so_far events, insertions into the open queue, etc.

    Args:
        :param lite.Connection con: DB connection
        :param str operation: result, current, prom_O, skip
        :param str hname: hash name, same as MD filenames
        :param str src: from WQ (open queue)
        :param str dst: to VIZ (visited)
        :param list bsf: all best_so_far values for each metric
        :param int gc: greedy counter - affects events like seed change
        :param float mul: greedy multiplier - controls greediness
        :param list prev_arr: distance from the previous node
        :param list goal_arr: distance to the goal
        :param str cur_metr_name: name of the current metric

    Returns:
    Stores data in the DB in a log table.
    """
    src = 'None' if src == '' else src
    dst = 'None' if dst == '' else dst
    nid = get_id_for_hash(con, hname)
    nid = 'None' if nid is None else nid
    columns = 'operation, id, src, dst, cur_metr, bsfr, bsfrb, bsfn, bsfh, bsfa, bsfx, gc, mul, '

    if not isinstance(goal_arr, (list,)):  # short version for skip operation
        columns += 'dist_from_prev_1, dist_to_goal_1'
        final_str = ', '.join('"{}"'.format(elem) if isinstance(elem, str) else str(elem)
                              for elem in (operation, nid, src, dst, cur_metr_name, bsf["BBRMSD"], bsf["AARMSD"], bsf["ANGL"],
                                           bsf["AND_H"], bsf["AND"], bsf["XOR"], gc, mul, prev_arr, goal_arr))
    else:
        nseeds = len(prev_arr)  # long version for append operation
        columns += ', '.join(('dist_from_prev_{0}'.format(i+1) for i in range(nseeds))) + ', '
        columns += ', '.join(('dist_to_goal_{0}'.format(i+1) for i in range(nseeds)))
        prev_arr_str = ', '.join((str(elem) for elem in prev_arr))
        goal_arr_str = ', '.join((str(elem) for elem in goal_arr))
        final_str = ', '.join('"{}"'.format(elem) if isinstance(elem, str) else str(elem)
                              for elem in (operation, nid, src, dst, cur_metr_name, bsf["BBRMSD"], bsf["AARMSD"], bsf["ANGL"],
                                           bsf["AND_H"], bsf["AND"], bsf["XOR"], gc, mul))
        final_str += ", ".join(('', prev_arr_str, goal_arr_str))

    qry = 'INSERT INTO log({}) VALUES ({})'.format(columns, final_str)
    cur = con.cursor()
    try:
        cur.execute(qry)
        con.commit()
    except Exception as e:
        print(e, '\nqry: ', operation, hname, src, dst, bsf, gc, mul, prev_arr, goal_arr)
        print('Extra info: ', qry)
        print('Type of function : {}'.format('Short' if not isinstance(goal_arr, (list,)) else 'Long'))
        log_error(con, 'LOG', nid)


# def prep_insert_into_log(con, operation, name, src, dst, bsf, gc, mul, prev_arr, goal_arr):
#     src = 'None' if src == '' else src
#     nid = get_id_for_name(con, name)
#     columns = 'operation, id, src, dst, bsf, gc, mul, '
#
#     if isinstance(goal_arr, (float, int)):  # short version
#         columns += 'dist_from_prev_1, dist_to_goal_1'
#         final_str = ', '.join('"{}"'.format(elem) if isinstance(elem, str) else str(elem)
#                               for elem in (operation, nid, src, dst, bsf, gc, mul, prev_arr, goal_arr))
#     else:
#         nseeds = len(prev_arr)
#         columns += ', '.join(('dist_from_prev_{0}, dist_to_goal_{0}'.format(i+1) for i in range(nseeds)))
#         prev_arr_str = ', '.join((str(elem) for elem in prev_arr))
#         goal_arr_str = ', '.join((str(elem) for elem in goal_arr))
#         final_str = ', '.join('"{}"'.format(elem) if isinstance(elem, str) else str(elem)
#                               for elem in (operation, nid, src, dst, bsf, gc, mul))
#         final_str += ", ".join(('', prev_arr_str, goal_arr_str))
#
#     return final_str


def copy_old_db(main_dict_keys: list, last_visited: list, next_in_oq: str, last_gc: float) -> NoReturn:
    """Used during the recovery procedure.

    Args:
        :param list main_dict_keys: all hash values from the main_dict - storage of all metric information
        :param list last_visited: several (3) recent values from the visited queue
        :param str next_in_oq: next hash (id) in the open queue, used for double check
        :param float last_gc: last greedy counter observed in the information from the pickle

    Returns:
    Conditionally copies data from the previous DB into a new one as a part of the restore process.
    """
    counter = 0
    db_path = os.getcwd()
    # db_name = 'results_{}.sqlite3'.format(counter)
    full_path = os.path.join(db_path, 'results_{}.sqlite3'.format(counter))

    while os.path.exists(full_path):
        prev_db = full_path
        counter += 1
        full_path = os.path.join(db_path, 'results_{}.sqlite3'.format(counter))

    #  yes, prev_db - the last one which exists
    cur_con = lite.connect(prev_db, check_same_thread=False, isolation_level=None)

    current_db_cur = cur_con.cursor()

    current_db_cur.execute("DELETE FROM log")
    current_db_cur.execute("DELETE FROM visited")
    current_db_cur.execute("DELETE FROM main_storage")
    cur_con.commit()

    prev_db_con = lite.connect(os.path.join(db_path, 'results_{}.sqlite3'.format(counter - 2)), check_same_thread=False, isolation_level=None)

    hashes = get_all_hashed_names(prev_db_con)
    for hash_hame in hashes:
        if hash_hame[0] in main_dict_keys:
            break

    max_id = get_id_for_hash(prev_db_con, hash_hame[0])
    prev_ids = [get_id_for_hash(prev_db_con, last_visited[0][2]), get_id_for_hash(prev_db_con, last_visited[1][2]), get_id_for_hash(prev_db_con, last_visited[2][2])]
    next_id = get_id_for_hash(prev_db_con, next_in_oq)
    # del last_visited, next_in_oq
    max_vid, vid_ts, last_vis_id = get_corr_vid_for_id(prev_db_con, max_id, prev_ids, last_gc)
    max_lid = get_corr_lid_for_id(prev_db_con, next_id, vid_ts, last_vis_id)

    prev_db_con.close()
    del prev_db_con, hash_hame, hashes, main_dict_keys

    current_db_cur.execute("ATTACH DATABASE ? AS prev_db", ('results_{}.sqlite3'.format(counter-2),))  # -1 - cur, -2 - prev

    current_db_cur.execute("INSERT INTO main.main_storage SELECT * FROM prev_db.main_storage WHERE prev_db.main_storage.id <= ?", (max_id,))
    cur_con.commit()
    current_db_cur.execute("INSERT INTO main.visited SELECT * FROM prev_db.visited WHERE prev_db.visited.vid <= ?", (max_vid,))
    cur_con.commit()
    current_db_cur.execute("INSERT INTO main.log SELECT * FROM prev_db.log WHERE prev_db.log.lid <= ?", (max_lid,))
    cur_con.commit()

#
# def sync_state_with_db(state):
#     counter = 0
#     db_path = os.getcwd()
#     db_name = 'results_{}.sqlite3'.format(counter)
#     full_path = os.path.join(db_path, 'results_{}.sqlite3'.format(counter))
#
#     while os.path.exists(full_path):
#         prev_db = full_path
#         counter += 1
#         full_path = os.path.join(db_path, 'results_{}.sqlite3'.format(counter))
#
#     #  yes, prev_db - last one which exists
#     cur_con = lite.connect(prev_db, check_same_thread=False, isolation_level=None)
#
#     current_db_cur = cur_con.cursor()
#
#     current_db_cur.execute("DELETE FROM log")
#     # get_conn
#     # get indexes
#     # drop all log with
#     # drop all vis with
#     # drop all main with
#     # vacuum
#     return True