def parse_top_for_h(top_filename: str) -> list:
    """
    Reads the topology file and finds positions of the hydrogen atoms
    :param top_filename: topology file .top
    :return: list of hydrogen atoms position
    :rtype: list
    """
    good_ind = list()
    with open(top_filename, 'r') as f:
        line = f.readline()
        while '[ atoms ]' not in line:
            line = f.readline()
        line = f.readline()
        atom_ind = line[1:].strip().split().index('atom')
        while ';' == line[0]:
            line = f.readline()
        line = line.strip()
        while len(line):
            if line[0] != ';':
                parsed_line = line.split(';')[0].split()
                if parsed_line[atom_ind][0] == 'H':
                    good_ind.append(int(parsed_line[0]))
                    # good_ind.append(int(parsed_line[0]) - 1)  # -1 for corr indexing
            line = f.readline().strip()
    return good_ind


# parse_top_for_h('./prot_dir/topol.top')