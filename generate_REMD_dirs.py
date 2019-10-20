#!/usr/bin/env python3

import os
from shutil import copy2 as cp2


def gen_dirs():
    root_dir = 'REMD_profiles'
    cur_prot = 'TRP'
    tot_steps = 31250000  # trp 100 000
    # tot_steps = 166670000  # vil 100 000
    # tot_steps = 250000000  # gb1 800 000

    full_path = os.path.join(root_dir, cur_prot)
    ffs = ['amber', 'charm', 'gromos', 'opls']

    trp_profile_1 = [300.00, 302.87, 305.77, 308.69, 311.63, 314.59, 317.57, 320.58, 323.62, 326.67, 329.75, 332.86, 335.98, 339.13, 342.31,
                345.51, 348.74, 351.99, 355.26, 358.56, 361.90, 365.25, 368.63, 372.04, 375.48, 378.93, 382.42, 385.94, 389.48, 393.05,
                396.65, 400.00]  # amber, charm, opls
    trp_profile_2 = [300.00, 302.90, 305.83, 308.78, 311.76, 314.76, 317.78, 320.82, 323.89, 326.98, 330.10, 333.25, 336.41, 339.61, 342.82,
                346.07, 349.34, 352.63, 355.95, 359.30, 362.67, 366.07, 369.50, 372.94, 376.42, 379.92, 383.46, 387.02, 390.62, 394.23,
                397.89, 400.00]  # gromos

    vil_profile_1 = [300.00, 303.07, 306.17, 309.30, 312.46, 315.64,
318.85, 322.09, 325.35, 328.63, 331.95, 335.28, 338.65, 342.05, 345.48, 348.93,
352.42, 355.93, 359.48, 363.05, 366.65, 370.29, 373.95, 377.64, 381.37, 385.13,
388.91, 392.73, 396.59, 400.00
]  # amber, charm, opls
    vil_profile_2 = [300.00, 303.15, 306.32, 309.52, 312.75, 316.01, 319.29,
322.58, 325.92, 329.29, 332.68, 336.11, 339.57, 343.05, 346.57, 350.11, 353.69,
357.29, 360.93, 364.59, 368.29, 372.02, 375.79, 379.58, 383.41, 387.27, 391.17,
395.10, 399.06, 400.00
]  # gromos

    gb1_profile_1 = [300.00, 302.57, 305.16, 307.76, 310.39, 313.03,
                    315.69, 318.37, 321.07, 323.78, 326.52, 329.27, 332.05, 334.84, 337.62, 340.45,
                    343.30, 346.17, 349.07, 351.98, 354.91, 357.86, 360.84, 363.83, 366.84, 369.88,
                    372.94, 376.01, 379.11, 382.22, 385.37, 388.53, 391.72, 394.93, 398.16, 400.00
]  # amber, charm, opls
    gb1_profile_2 = [300.00, 302.57, 305.15, 307.76, 310.38, 313.03, 315.69,
                    318.37, 321.07, 323.78, 326.52, 329.27, 332.05, 334.84, 337.62, 340.45, 343.30,
                    346.17, 349.06, 351.98, 354.91, 357.86, 360.84, 363.83, 366.84, 369.88, 372.94,
                    376.01, 379.11, 382.23, 385.37, 388.54, 391.70, 394.91, 398.14, 400.00
]  # gromos

    profile_1 = trp_profile_1
    profile_2 = trp_profile_2

    temperartures = [
        profile_1,
        profile_1,
        profile_2,
        profile_1
    ]

    try:
        os.mkdir(root_dir)
    except:
        print('Failed to create directory {}.'.format(root_dir))

    try:
        os.mkdir(full_path)
    except:
        print('Failed to create directory {}.'.format(full_path))

    gpu_flag = True

    for i, ff in enumerate(ffs):
        work_dir = os.path.join(full_path, ff)
        try:
            os.mkdir(work_dir)
        except:
            print('Failed to create directory {}.'.format(os.path.join(full_path, ff)))
        for j, temp in enumerate(temperartures[i]):
            if gpu_flag:
                mdp_content = get_mdp_str_gpu(name='REMD {}@{}'.format(cur_prot, ff), temp=temp, seed=1, steps=tot_steps)
            else:
                mdp_content = get_mdp_str_ener_gr(name='REMD {}@{}'.format(cur_prot, ff), temp=temp, seed=1, steps=tot_steps)
            temp_dir = os.path.join(work_dir, '{}_{}_{}'.format(cur_prot, ff, j+1))
            try:
                os.mkdir(temp_dir)
            except:
                pass
            with open(os.path.join(temp_dir, 'md.mdp'), 'w') as mdp_file:
                mdp_file.write(mdp_content)

        # cp2(os.path.join(conf_files_dir, 'prot.ndx'), work_dir)
        # if ff == 'charm':
        #     cp2(os.path.join(conf_files_dir, 'charmm36-nov2018.ff'), work_dir)


def get_mdp_str_ener_gr(name: str, temp: float, seed: int, steps: int):
    """

        :param str name:
        :param float temp:
        :param int seed:
        :param int steps:
    """
    mdp_str = "\
        ; Run parameters\n\
        integrator	= md		; leap-frog integrator\n\
        nsteps		= {3:d}		; 2 * 10000 = 20 ps\n\
        dt		    = 0.002		; 2 fs\n\
        ld-seed     = {2:d}     ; \n\
        ; Output control\n\
        nstxout		= 0  	; save coordinates every 0.0 ps\n\
        nstvout		= 0		; save velocities every 0.0 ps\n\
        nstenergy	= 10000 ; save energies every 0.0 ps\n\
        nstlog		= 10000		; update log file every 0.0 ps\n\
        nstxout-compressed	= 10000	; save coordinates every 0.0 ps\n\
        energygrps  = Protein SOL\n\
        ; Bond parameters\n\
        continuation	        = no		; first dynamics run\n\
        constraint_algorithm    = lincs	    ; holonomic constraints \n\
        constraints	            = h-bonds	; all bonds (even heavy atom-H bonds) constrained\n\
        lincs_iter	            = 1		    ; accuracy of LINCS\n\
        lincs_order	            = 4		    ; also related to accuracy\n\
        ; Neighborsearching\n\
        cutoff-scheme   = Verlet\n\
        ns_type		    = grid		; search neighboring grid cells\n\
        nstlist		    = 10		; 20 fs, largely irrelevant with Verlet\n\
        rcoulomb	    = 1.0		; short-range electrostatic cutoff (in nm)\n\
        rvdw		    = 1.0		; short-range van der Waals cutoff (in nm)\n\
        ; Electrostatics\n\
        coulombtype	    = PME	; Particle Mesh Ewald for long-range electrostatics\n\
        pme_order	    = 4		; cubic interpolation\n\
        fourierspacing	= 0.16	; grid spacing for FFT\n\
        ; Temperature coupling is on\n\
        tcoupl		= V-rescale	            ; modified Berendsen thermostat\n\
        tc-grps		= Protein Non-Protein	; two coupling groups - more accurate\n\
        tau_t		= 0.1	  0.1           ; time constant, in ps\n\
        ref_t		= {1:f} 	  {1:f}           ; reference temperature, one for each group, in K\n\
        ; Pressure coupling is off\n\
        pcoupl		= no 		; no pressure coupling in NVT\n\
        ; Periodic boundary conditions\n\
        pbc		= xyz		    ; 3-D PBC\n\
        ; Dispersion correction\n\
        DispCorr	= EnerPres	; account for cut-off vdW scheme\n\
        ; Velocity generation\n\
        gen-vel		= yes		; assign velocities from Maxwell distribution\n\
        gen-temp	= {1:f}		; temperature for Maxwell distribution\n\
        gen-seed	= {2:d}		; generate a random seed".format(name, temp, seed, steps)
    return mdp_str


def get_mdp_str_gpu(name: str, temp: float, seed: int, steps: int):
    """

        :param str name:
        :param float temp:
        :param int seed:
        :param int steps:
    """
    mdp_str = "\
        ; Run parameters\n\
        integrator	= md		; leap-frog integrator\n\
        nsteps		= {3:d}		; 2 * 10000 = 20 ps\n\
        dt		    = 0.002		; 2 fs\n\
        ld-seed     = {2:d}     ; \n\
        ; Output control\n\
        nstxout		= 0  	; save coordinates every 0.0 ps\n\
        nstvout		= 0		; save velocities every 0.0 ps\n\
        nstenergy	= 0 ; save energies every 0.0 ps\n\
        nstlog		= 10000		; update log file every 0.0 ps\n\
        nstxout-compressed	= 10000	; save coordinates every 0.0 ps\n\
        ; Bond parameters\n\
        continuation	        = no		; first dynamics run\n\
        constraint_algorithm    = lincs	    ; holonomic constraints \n\
        constraints	            = h-bonds	; all bonds (even heavy atom-H bonds) constrained\n\
        lincs_iter	            = 1		    ; accuracy of LINCS\n\
        lincs_order	            = 4		    ; also related to accuracy\n\
        ; Neighborsearching\n\
        cutoff-scheme   = Verlet\n\
        ns_type		    = grid		; search neighboring grid cells\n\
        nstlist		    = 10		; 20 fs, largely irrelevant with Verlet\n\
        rcoulomb	    = 1.0		; short-range electrostatic cutoff (in nm)\n\
        rvdw		    = 1.0		; short-range van der Waals cutoff (in nm)\n\
        ; Electrostatics\n\
        coulombtype	    = PME	; Particle Mesh Ewald for long-range electrostatics\n\
        pme_order	    = 4		; cubic interpolation\n\
        fourierspacing	= 0.16	; grid spacing for FFT\n\
        ; Temperature coupling is on\n\
        tcoupl		= V-rescale	            ; modified Berendsen thermostat\n\
        tc-grps		= Protein Non-Protein	; two coupling groups - more accurate\n\
        tau_t		= 0.1	  0.1           ; time constant, in ps\n\
        ref_t		= {1:f} 	  {1:f}           ; reference temperature, one for each group, in K\n\
        ; Pressure coupling is off\n\
        pcoupl		= no 		; no pressure coupling in NVT\n\
        ; Periodic boundary conditions\n\
        pbc		= xyz		    ; 3-D PBC\n\
        ; Dispersion correction\n\
        DispCorr	= EnerPres	; account for cut-off vdW scheme\n\
        ; Velocity generation\n\
        gen-vel		= yes		; assign velocities from Maxwell distribution\n\
        gen-temp	= {1:f}		; temperature for Maxwell distribution\n\
        gen-seed	= {2:d}		; generate a random seed".format(name, temp, seed, steps)
    return mdp_str


if __name__ == '__main__':
    gen_dirs()