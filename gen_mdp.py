
def get_mdp(seed: int, temp: int, name: str = 'default') -> str:
    """
    Generates text for .mdf file with simulation settings
    :param seed: seed to be used for initial velocities generation
    :param temp: temperature of the experiment
    :param name: name of the experiment inside the .mdp file
    :return: string with .mdp text
    :rtype: str
    """
    calibration_mdp = "\
; Run parameters\n\
integrator	= md		; leap-frog integrator\n\
nsteps		= 10000		; 2 * 10000 = 20 ps\n\
dt		    = 0.002		; 2 fs\n\
ld-seed     = {2:d}     ; \n\
; Output control\n\
nstxout		= 0  	; save coordinates every 0.0 ps\n\
nstvout		= 0		; save velocities every 0.0 ps\n\
nstenergy	= 10000 ; save energies every 0.0 ps\n\
nstlog		= 0		; update log file every 0.0 ps\n\
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
ref_t		= {1:d} 	  {1:d}           ; reference temperature, one for each group, in K\n\
; Pressure coupling is off\n\
pcoupl		= no 		; no pressure coupling in NVT\n\
; Periodic boundary conditions\n\
pbc		= xyz		    ; 3-D PBC\n\
; Dispersion correction\n\
DispCorr	= EnerPres	; account for cut-off vdW scheme\n\
; Velocity generation\n\
gen-vel		= yes		; assign velocities from Maxwell distribution\n\
gen-temp	= {1:d}		; temperature for Maxwell distribution\n\
gen-seed	= {2:d}		; generate a random seed".format(name, temp, seed)
    return calibration_mdp