#!/usr/bin/env python3
from parse_topology_for_hydrogens import parse_top_for_h


def gen_h_ndx(orig_ndx, topology, out_name='h_prot.ndx'):

    ndx_ind = list()
    with open(orig_ndx, 'r') as f:
        line = f.readline()
        while '[ Protein ]' not in line:
            line = f.readline()
        line = f.readline()
        while ';' == line[0]:
            line = f.readline()
        line = line.strip()
        while len(line):
            ndx_ind.extend(line.split())
            line = f.readline().strip()
    ndx_ind = [int(elem) for elem in ndx_ind]

    good_ind = parse_top_for_h(topology)
    filtered_h_ind = [elem for elem in ndx_ind if elem in good_ind]
    formated_h_ind = ['{:>4} '.format(elem) for elem in filtered_h_ind]
    with open(out_name, 'w') as new_file:
        ind = 0
        new_file.write('[ Protein ]\n')
        while ind < len(filtered_h_ind):
            new_file.write(''.join(formated_h_ind[ind:ind+15]))
            new_file.write('\n')
            # print(''.join(formated_h_ind[ind:ind+15]))
            ind += 15


# gen_h_ndx('./prot_dir/prot.ndx', './prot_dir/topol.top')
