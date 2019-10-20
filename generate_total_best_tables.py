#!/usr/bin/env python3

import os
import sqlite3 as lite
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import figaspect
import multiprocessing as mp

# #########   TRP   ######################
# for ff in ffs:
#     filenames_db = ['results_{}_trp_300_fixed.sqlite3'.format(ff), 'results_{}_trp_300_2_fixed.sqlite3'.format(ff)]
#     legend_names = ['TRP {}_1'.format(ff), 'TRP {}_2'.format(ff)]
#     common_path = '../trp_{}_compar'.format(ff)
#     batch_arr.append((filenames_db, legend_names, common_path))
#
# filenames_db = ['results_amber_trp_300_2_fixed.sqlite3', 'results_charm_trp_300_2_fixed.sqlite3', 'results_gromos_trp_300_2_fixed.sqlite3', 'results_opls_trp_300_2_fixed.sqlite3']
# legend_names = ['TRP amber_2', 'TRP charm_2', 'TRP gromos_2', 'TRP opls_2']
# common_path = '../trp_all_2_compar'
# batch_arr.append((filenames_db, legend_names, common_path))
#
# filenames_db = ['results_amber_trp_300_fixed.sqlite3', 'results_charm_trp_300_fixed.sqlite3', 'results_gromos_trp_300_fixed.sqlite3', 'results_opls_trp_300_fixed.sqlite3']
# legend_names = ['TRP amber', 'TRP charm', 'TRP gromos', 'TRP opls']
# common_path = '../trp_all_1_compar'
# batch_arr.append((filenames_db, legend_names, common_path))
#
# filenames_db = ['results_amber_trp_300_fixed.sqlite3', 'results_amber_trp_300_2_fixed.sqlite3', 'results_charm_trp_300_fixed.sqlite3', 'results_charm_trp_300_2_fixed.sqlite3', 'results_gromos_trp_300_fixed.sqlite3', 'results_gromos_trp_300_2_fixed.sqlite3', 'results_opls_trp_300_fixed.sqlite3', 'results_opls_trp_300_2_fixed.sqlite3']
# legend_names = ['TRP amber', 'TRP amber_2', 'TRP charm', 'TRP charm_2', 'TRP gromos', 'TRP gromos_2', 'TRP opls', 'TRP opls_2']
# common_path = '../trp_all_compar'
# batch_arr.append((filenames_db, legend_names, common_path))

# ##################  VIL  #######################
#
# filenames_db = ['results_amber_vil_300.sqlite3', 'results_charm_vil_300.sqlite3', 'results_gromos_vil_300.sqlite3', 'results_opls_vil_300.sqlite3']
# legend_names = ['VIL amber', 'VIL charm', 'VIL gromos', 'VIL opls']
# common_path = '../vil_all_compar'
# batch_arr.append((filenames_db, legend_names, common_path))


# ##################  GB1  #######################
#
# filenames_db = ['results_amber_gb1_300.sqlite3', 'results_charm_gb1_300.sqlite3', 'results_gromos_gb1_300.sqlite3', 'results_opls_gb1_300.sqlite3']
# legend_names = ['GB1s amber', 'GB1 charm', 'GB1 gromos', 'GB1 opls']
# common_path = '../gb1_all_compar'
# batch_arr.append((filenames_db, legend_names, common_path))



def main():

    # ############ TRP #########
    # filenames_db = ['results_amber_trp_300_fixed.sqlite3', 'results_amber_trp_300_2_fixed.sqlite3', 'results_charm_trp_300_fixed.sqlite3', 'results_charm_trp_300_2_fixed.sqlite3', 'results_gromos_trp_300_fixed.sqlite3', 'results_gromos_trp_300_2_fixed.sqlite3', 'results_opls_trp_300_fixed.sqlite3', 'results_opls_trp_300_2_fixed.sqlite3']
    # table_names = ['amber trp 1', 'amber trp 2', 'charm trp 1', 'charm trp 2', 'gromos trp 1', 'gromos trp 2', 'opls trp 1', 'opls trp 2']
    # outfile = 'all_trp_all'
    # plot_tables(filenames_db, outfile, table_names)

    # filenames_db = ['results_amber_trp_300_fixed.sqlite3', 'results_charm_trp_300_fixed.sqlite3', 'results_gromos_trp_300_fixed.sqlite3', 'results_opls_trp_300_fixed.sqlite3']
    # table_names = ['amber trp 1', 'charm trp 1', 'gromos trp 1', 'opls trp 1']
    # outfile = 'all_trp_1'
    # plot_tables(filenames_db, outfile, table_names)

    # filenames_db = ['results_amber_trp_300_2_fixed.sqlite3', 'results_charm_trp_300_2_fixed.sqlite3', 'results_gromos_trp_300_2_fixed.sqlite3', 'results_opls_trp_300_2_fixed.sqlite3']
    # table_names = ['amber trp 2', 'charm trp 2', 'gromos trp 2', 'opls trp 2']
    # outfile = 'all_trp_2'
    # plot_tables(filenames_db, outfile, table_names)

    # filenames_db = ['results_amber_trp_300_fixed.sqlite3', 'results_amber_trp_300_2_fixed.sqlite3']
    # table_names = ['amber trp 1', 'amber trp 2']
    # outfile = 'amber_trp'
    # plot_tables(filenames_db, outfile, table_names)

    # filenames_db = ['results_charm_trp_300_fixed.sqlite3', 'results_charm_trp_300_2_fixed.sqlite3']
    # table_names = ['charm trp 1', 'charm trp 2']
    # outfile = 'charm_trp'
    # plot_tables(filenames_db, outfile, table_names)

    # filenames_db = ['results_gromos_trp_300_fixed.sqlite3', 'results_gromos_trp_300_2_fixed.sqlite3']
    # table_names = ['gromos trp 1', 'gromos trp 2']
    # outfile = 'gromos_trp'
    # plot_tables(filenames_db, outfile, table_names)

    # filenames_db = ['results_opls_trp_300_fixed.sqlite3', 'results_opls_trp_300_2_fixed.sqlite3']
    # table_names = ['opls trp 1', 'opls trp 2']
    # outfile = 'opls_trp'
    # plot_tables(filenames_db, outfile, table_names)


    # # ############ VIL #########
    # filenames_db = ['results_amber_vil_300.sqlite3', 'results_charm_vil_300.sqlite3', 'results_gromos_vil_300.sqlite3', 'results_opls_vil_300.sqlite3']
    # table_names = ['amber vil', 'charm vil', 'gromos vil', 'opls vil']
    # outfile = 'all_vil'
    # plot_tables(filenames_db, outfile, table_names)


    # ############ GB1 #########
    filenames_db = ['results_amber_gb1_300.sqlite3', 'results_charm_gb1_300.sqlite3', 'results_gromos_gb1_300.sqlite3', 'results_opls_gb1_300.sqlite3']
    table_names = ['amber gb1', 'charm gb1', 'gromos gb1', 'opls gb1']
    outfile = 'all_gb1'
    plot_tables(filenames_db, outfile, table_names)


def plot_tables(filenames_db: list, out_file: str, table_names: list):
    """

    Args:
        :param list filenames_db:
        :param str out_file:
        :param list table_names:
    """
    out_file = '{}.tex'.format(out_file)
    con_arr = [lite.connect(db_name, check_same_thread=False, isolation_level=None) for db_name in filenames_db]
    cur_arr = [con.cursor() for con in con_arr]
    metrics = ["RMSD", "ANGL", "AND_H", "AND", "XOR"]
    metrics_tab = ["RMSD", "ANGL", "AND\\_H", "AND", "XOR"]
    allowed_faild = [20, 10, 5, 5, 10]

    total_promotions = list()
    prom_during_metric = list()
    total_steps_during_metric = list()
    for db_name in filenames_db:
        con = lite.connect(db_name, check_same_thread=False, isolation_level=None)
        cur = con.cursor()
        qry = "select count(1) from log where operation='prom_O' "  # total
        result = cur.execute(qry)
        total_promotions.append(result.fetchone()[0])
        personal_res = list()
        personal_total_steps = list()
        for partial_metr in metrics:
            qry = "select count(1) from log where operation='prom_O' and cur_metr='{}'".format(partial_metr)
            result = cur.execute(qry)
            personal_res.append(result.fetchone()[0])
        prom_during_metric.append(personal_res)
        for partial_metr in metrics:
            qry = "select count(1) from log where dst='VIZ' and cur_metr='{}'".format(partial_metr)
            result = cur.execute(qry)
            personal_total_steps.append(result.fetchone()[0])
        total_steps_during_metric.append(personal_total_steps)
        del personal_res
        con.close()
    del result, qry, partial_metr, db_name, cur, con, con_arr, cur_arr, personal_total_steps
    a = 8
    # for i in range(len(total_promotions)):
    #     print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format('metr', 'per_s', 'tot_s', 'pe/to', 'pe/al   ', 'to/al', 'to/al ', 'fa'))
    #     for j in range(len(prom_during_metric[i])):
    #         print('{}:\t{:d}  \t{:d}\t{:3.2f}\t{:3.2f} \t{:3.2f}\t{:3.2f}\t{:d}'.format(metrics[j], prom_during_metric[i][j], total_steps_during_metric[i][j], 100*prom_during_metric[i][j]/total_steps_during_metric[i][j], 100 * total_steps_during_metric[i][j] / total_promotions[i], 100 * prom_during_metric[i][j] / total_promotions[i], 100 * prom_during_metric[i][j] / sum(total_promotions), allowed_faild[j]))
    #     print('t: {}\t{}'.format(total_promotions[i], sum(total_steps_during_metric[i])))
    #     print()


    with open(out_file, 'w') as tex_table:
        tex_table.writelines(['\\begin{table}[h]\n', '\centering\n', '\sisetup{table-align-text-post=false}\n', '\\begin{tabular}{@{}|l|S[table-format=2.0]\
                                                                                                                                        |S[table-format=3.0]\
                                                                                                                                        |S[table-format=6]\
                                                                                                                                        |S[table-format=3.3]\
                                                                                                                                        |S[table-format=3.2]\
                                                                                                                                        |S[table-format=3.3]\
                                                                                                                                        |S[table-format=1.2]\
                                                                                                                                        |@{}}\n'])
        for i in range(len(total_promotions)):
            tex_table.write('')

            tex_table.write('\multicolumn{{8}}{{c}}{{{}}} \\\\ \n'.format(table_names[i]))
            tex_table.write('\\hline\n')
            tex_table.write('{} & {} & {} & {} & {} & {} & {} & {} \\\\ \n'.format(          '',      '{allowed}', '{percent}', '{metric}',   '{percent}', '{promotions}', '{percent of}', '{promotions}'))
            tex_table.write('{} & {} & {} & {} & {} & {} & {} & {} \\\\ \\hline\n'.format('{metric}', '{fails}', '{allowed}', '{total steps}', '{steps}', '{per metric}', '{promotions}', '{per 1000 steps}'))
            for j in range(len(prom_during_metric[i])):
                tex_table.write('{:s} & {:d} & {:3.0f}\\si{{\\percent}} & {:d} & {:3.2f}\\si{{\\percent}} & {} & {:3.2f}\\si{{\\percent}}  & {:3.2f} \\\\ \\hline\n'.format(
                    metrics_tab[j],
                    allowed_faild[j],
                    100*allowed_faild[j]/sum(allowed_faild),
                    total_steps_during_metric[i][j],
                    100*total_steps_during_metric[i][j]/sum(total_steps_during_metric[i]),
                    prom_during_metric[i][j],
                    100 * prom_during_metric[i][j]/sum(prom_during_metric[i]),
                    1000 * prom_during_metric[i][j]/total_steps_during_metric[i][j]))
            tex_table.write('{:s} & {:d} & {:3.0f}\\si{{\\percent}} & {:d} & {:3.2f}\\si{{\\percent}} & {} & {:3.2f}\\si{{\\percent}} & {:3.2f}\\\\ \\hline \\hline \n'.format('total', sum(allowed_faild), 100, sum(total_steps_during_metric[i]), 100, sum(prom_during_metric[i]), 100, 1000 * sum(prom_during_metric[i])/sum(total_steps_during_metric[i])))
        tex_table.writelines(['\\end {tabular}\n', '\\caption{{{}}}\n'.format('{}'.format(', '.join(table_names))), '\\end {table}\n'])

        tex_table.write('\n\n\n')


        total_steps_during_metric_comb = [sum(x) for x in zip(*total_steps_during_metric)]
        prom_during_metric_comb = [sum(x) for x in zip(*prom_during_metric)]

        tex_table.writelines(['\\begin{table}[h]\n', '\centering\n', '\sisetup{table-align-text-post=false}\n', '\\begin{tabular}{@{}|l|S[table-format=2.0]\
                                                                                                                                        |S[table-format=3.0]\
                                                                                                                                        |S[table-format=6]\
                                                                                                                                        |S[table-format=3.3]\
                                                                                                                                        |S[table-format=3.2]\
                                                                                                                                        |S[table-format=3.3]\
                                                                                                                                        |S[table-format=1.2]\
                                                                                                                                        |@{}}\n', '\\hline\n'])
        tex_table.write('{} & {} & {} & {} & {} & {} & {} & {} \\\\ \n'.format('', '{allowed}', '{percent}', '{metric}', '{percent}', '{promotions}', '{percent of}', '{promotions}'))
        tex_table.write('{} & {} & {} & {} & {} & {} & {} & {} \\\\ \\hline\n'.format('{metric}', '{fails}', '{allowed}', '{total steps}', '{steps}', '{per metric}', '{promotions}', '{per 1000 steps}'))
        for j in range(len(prom_during_metric_comb)):
            tex_table.write('{:s} & {:d} & {:3.0f}\\si{{\\percent}} & {:d} & {:3.2f}\\si{{\\percent}} & {} & {:3.2f}\\si{{\\percent}}  & {:3.2f} \\\\ \\hline\n'.format(
                metrics_tab[j],
                allowed_faild[j],
                100*allowed_faild[j]/sum(allowed_faild),
                total_steps_during_metric_comb[j],
                100*total_steps_during_metric_comb[j]/sum(total_steps_during_metric_comb),
                prom_during_metric_comb[j],
                100 * prom_during_metric_comb[j]/sum(prom_during_metric_comb),
                1000 * prom_during_metric_comb[j]/total_steps_during_metric_comb[j]))
        tex_table.write('{:s} & {:d} & {:3.0f}\\si{{\\percent}} & {:d} & {:3.2f}\\si{{\\percent}} & {} & {:3.2f}\\si{{\\percent}} & {:3.2f}\\\\ \\hline \\hline \n'.format('total', sum(allowed_faild), 100, sum(total_steps_during_metric_comb), 100, sum(prom_during_metric_comb), 100, 1000 * sum(prom_during_metric_comb)/sum(total_steps_during_metric_comb)))
        tex_table.writelines(['\\end {tabular}\n', '\\caption{{{}}}\n'.format('Summary of ({})'.format(', '.join(table_names))), '\\end {table}\n'])



        tex_table.write('\n\n\n')
        norm_coef = [min(allowed_faild)/elem   for elem in allowed_faild]
        allowed_faild = [elem * norm_coef[k] for k, elem in enumerate(allowed_faild)]

        tex_table.writelines(['\\begin{table}[h]\n', '\centering\n', '\sisetup{table-align-text-post=false}\n', '\\begin{tabular}{@{}|l|S[table-format=2.0]\
                                                                                                                                        |S[table-format=3.0]\
                                                                                                                                        |S[table-format=6]\
                                                                                                                                        |S[table-format=3.3]\
                                                                                                                                        |S[table-format=3.2]\
                                                                                                                                        |S[table-format=3.3]\
                                                                                                                                        |S[table-format=1.2]\
                                                                                                                                        |@{}}\n'])

        for i in range(len(total_promotions)):
            total_steps_during_metric[i] = [elem * norm_coef[k] for k, elem in enumerate(total_steps_during_metric[i])]
            prom_during_metric[i] = [elem * norm_coef[k] for k, elem in enumerate(prom_during_metric[i])]

            tex_table.write('\multicolumn{{8}}{{c}}{{{}}} \\\\ \n'.format(table_names[i]))
            tex_table.write('\\hline\n')
            tex_table.write('{} & {} & {} & {} & {} & {} & {} & {} \\\\ \n'.format('', '{allowed}', '{percent}', '{metric}', '{percent}', '{promotions}', '{percent of}', '{promotions}'))
            tex_table.write(
                '{} & {} & {} & {} & {} & {} & {} & {} \\\\ \\hline\n'.format('{metric}', '{fails}', '{allowed}', '{total steps}', '{steps}', '{per metric}', '{promotions}', '{per 1000 steps}'))
            for j in range(len(prom_during_metric[i])):
                tex_table.write('{:s} & {:3.0f} & {:2.0f}\\si{{\\percent}} & {:3.0f} & {:3.2f}\\si{{\\percent}} & {} & {:3.2f}\\si{{\\percent}} & {:3.2f}  \\\\ \\hline\n'.format(
                    metrics_tab[j],
                    allowed_faild[j],
                    100*allowed_faild[j]/sum(allowed_faild),
                    total_steps_during_metric[i][j],
                    100*total_steps_during_metric[i][j]/sum(total_steps_during_metric[i]),
                    prom_during_metric[i][j],
                    100 * prom_during_metric[i][j]/sum(prom_during_metric[i]),
                    1000 * prom_during_metric[i][j]/total_steps_during_metric[i][j]))
            tex_table.write('{:s} & {:3.0f} & {:2.0f}\\si{{\\percent}} & {:2.0f} & {}\\si{{\\percent}} & {} & {}\\si{{\\percent}} & {:3.2f}\\\\ \\hline \\hline \n'.format('total', sum(allowed_faild), 100, sum(total_steps_during_metric[i]), 100, sum(prom_during_metric[i]), 100, 1000 * sum(prom_during_metric[i])/sum(total_steps_during_metric[i])))
        tex_table.writelines(['\\end {tabular}\n', '\\caption{{{}}}\n'.format('Normalized ' + '{}'.format(', '.join(table_names))), '\\end {table}\n'])

        total_steps_during_metric_comb = [sum(x) for x in zip(*total_steps_during_metric)]
        prom_during_metric_comb = [sum(x) for x in zip(*prom_during_metric)]

        tex_table.write('\n\n\n')

        tex_table.writelines(['\\begin{table}[h]\n', '\centering\n', '\sisetup{table-align-text-post=false}\n', '\\begin{tabular}{@{}|l|S[table-format=2.0]\
                                                                                                                                    |S[table-format=3.0]\
                                                                                                                                    |S[table-format=6]\
                                                                                                                                    |S[table-format=3.3]\
                                                                                                                                    |S[table-format=3.2]\
                                                                                                                                    |S[table-format=3.3]\
                                                                                                                                    |S[table-format=1.2]\
                                                                                                                                    |@{}}\n', '\\hline\n'])
        tex_table.write('{} & {} & {} & {} & {} & {} & {} & {} \\\\ \n'.format('', '{allowed}', '{percent}', '{metric}', '{percent}', '{promotions}', '{percent of}', '{promotions}'))
        tex_table.write('{} & {} & {} & {} & {} & {} & {} & {} \\\\ \\hline\n'.format('{metric}', '{fails}', '{allowed}', '{total steps}', '{steps}', '{per metric}', '{promotions}', '{per 1000 steps}'))
        for j in range(len(prom_during_metric_comb)):
            tex_table.write('{:s} & {:3.0f} & {:2.0f}\\si{{\\percent}} & {:3.0f} & {:3.2f}\\si{{\\percent}} & {} & {:3.2f}\\si{{\\percent}} & {:3.2f}  \\\\ \\hline\n'.format(
                metrics_tab[j],
                allowed_faild[j],
                100*allowed_faild[j]/sum(allowed_faild),
                total_steps_during_metric_comb[j],
                100*total_steps_during_metric_comb[j]/sum(total_steps_during_metric_comb),
                prom_during_metric_comb[j],
                100 * prom_during_metric_comb[j]/sum(prom_during_metric_comb),
                1000 * prom_during_metric_comb[j]/total_steps_during_metric_comb[j]))
        tex_table.write('{:s} & {:3.0f} & {:2.0f}\\si{{\\percent}} & {:2.0f} & {}\\si{{\\percent}} & {} & {}\\si{{\\percent}} & {:3.2f}\\\\ \\hline \\hline \n'.format('total', sum(allowed_faild), 100, sum(total_steps_during_metric_comb), 100, sum(prom_during_metric_comb), 100, 1000 * sum(prom_during_metric_comb)/sum(total_steps_during_metric_comb)))
        tex_table.writelines(['\\end {tabular}\n', '\\caption{{{}}}\n'.format('Normalized ' + 'summary of ({})'.format(', '.join(table_names))), '\\end {table}\n'])




        #     tex_table.writelin('\\hline')
        #     qry = "select count(1) from log where operation='prom_O' "
        #     result_arr = cur.execute(qry) cur_arr
        #     total_prom = [res.fetchone() for res in result_arr]
        #     for partial_metr in ["RMSD", "ANGL", "AND_H", "AND", "XOR"]:
        #         qry = "select count(1) from log where operation='prom_O' and cur_metr='{}'".format(partial_metr)
        #         result_arr = [cur.execute(qry) for cur in cur_arr]
        #         fetched_one_arr = [res.fetchone() for res in result_arr]
        #         tex_table.writelin('\\hline')
        #     tex_table.writelin('\\hline')
        #
        # tex_table.writelines(['\\caption\{{}\}'.format('some caption here'), '\\end {tabular}', '\\end {table}'])


if __name__ == '__main__':
    main()
