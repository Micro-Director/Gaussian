#-*-coding:utf-8-*-
import os
import sys
import numpy as np
import pandas as pd

from ase.symbols import Symbols

"""

Need add extract Force/Displacement module 

"""

log = sys.argv[1]
task = sys.argv[2]

class ExtactOptimizedStructure(object):

    def __init__(self, log_file,):
        
        self._log = log_file
        
        self._opted_structures = np.zeros(shape=(0,3), dtype=float)
        self._natoms = int(os.popen("grep -n 'NAtoms' " + self._log).read().split()[2])
        opt_line = np.array([os.popen("grep -n 'Standard orientation' " + self._log).read().split()]).reshape(-1, 3)[:,0]
        opt_line = [int(i[:-1]) for i in opt_line]
        if len(opt_line) > 1:
            itera_num = len(opt_line)-1
        else:
            itera_num = len(opt_line)
        for i_index, i in enumerate(opt_line[0:]):
            bk = np.array([os.popen("sed -n " + str(i+5) + ',' + str(i+4+self._natoms) + 'p ' + self._log).read().split()]).reshape(-1, 6).astype(np.float32)
            self._opted_structures = np.append(self._opted_structures, np.array(bk[:,-3:]), axis = 0)
        self._opted_structures.resize((itera_num, self._natoms, 3))
        sp = np.array(os.popen("sed -n " + str(opt_line[0]+5) + ',' + str(opt_line[0]+4+self._natoms) + 'p ' + self._log).read().split())
        self._species = [Symbols([i]).species().pop() for i in sp[1:][::6].astype(int)]
    
#    @property
    def final_xyz(self, info: str, dirr = None, file_name = None): 
        
        with open(dirr+ '/' + file_name + '.xyz', 'w+') as ods:

            ods.write(str(self._natoms) + '\n')
            ods.write(info + '\n')
            for i_index, i in enumerate(self._opted_structures[-1]):
                ods.write(self._species[i_index] + '   ' + str(round(i[0], 5)) + ' ' + str(round(i[1], 5)) + ' ' + str(round(i[2], 5)) + '\n')
            ods.write('\n')
    
    @property
    def get_coors(self):
        
        return self._opted_structures[-1]


def ExtactEnergyItera(log_file,):

    bk = os.popen("grep 'SCF Done' " + log_file).read().split()[4:][::9]
    energys = [float(i) for i in bk]
    
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 7))
    ax1 = plt.subplot(111)
    ax1.scatter(range(len(energys)), energys, s= 150, color = 'none', marker='o',edgecolors='darkblue', alpha = 0.8, linewidths=3)
    ax1.plot(range(len(energys)), energys, '-', color = 'darkblue', linewidth=3.0,)
    ax1.set_xlabel('Iteration step', fontsize = 25, fontweight = 'semibold')
    ax1.set_ylabel('Energy (eV)', fontsize = 25, fontweight = 'semibold', color = 'darkblue')
    ax1.tick_params(axis = 'both', labelsize = 25, width = 2, length = 8, direction='in')

    ax1.spines['bottom'].set_linewidth(3)
    ax1.spines['left'].set_linewidth(3) 
    ax1.spines['top'].set_linewidth(3)
    ax1.spines['right'].set_linewidth(3)        

    ax1.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig('energy.jpg', dpi = 300)
    
    return pd.DataFrame({'Step': range(len(energys)),
                         'Energy': energys})

def ExactOptCriitera(log_file):
    
    IteraLine = np.array(os.popen("grep -n ' Converged?' " + log_file).read().split('\n'))[:-1,]
    IteraLine = np.array([i.split(':')[0][:] for i in IteraLine]).astype(int)
    MaxForce  = np.zeros(shape=(0,), dtype=float)
    RMSForce = np.zeros(shape=(0, ), dtype=float)
    MaxDis = np.zeros(shape=(0, ), dtype = float)
    RMSDis = np.zeros(shape=(0, ), dtype= float)
    MFTh, RFTh, MDTh, RDTh = float(os.popen("sed -n " + str(IteraLine[0]+1) + "p " + log_file).read().split()[3]), float(os.popen("sed -n " + str(IteraLine[0]+2) + "p " + log_file).read().split()[3]), float(os.popen("sed -n " + str(IteraLine[0]+3) + "p " + log_file).read().split()[3]), float(os.popen("sed -n " + str(IteraLine[0]+4) + "p " + log_file).read().split()[3])
    
    for i in IteraLine: 
        MaxForce = np.append(MaxForce, float(os.popen("sed -n " + str(i + 1) + "p " + log_file).read().split()[2]), )
        RMSForce = np.append(RMSForce, float(os.popen("sed -n " + str(i + 2) + "p " + log_file).read().split()[2]), )
        MaxDis = np.append(MaxDis, float(os.popen("sed -n " + str(i + 3) + "p " + log_file).read().split()[2]), )
        RMSDis = np.append(RMSDis, float(os.popen("sed -n " + str(i + 4) + "p " + log_file).read().split()[2]), )
    
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (16, 14))
    ax1 = plt.subplot(221)
    ax1.scatter(range(len(MaxForce)), MaxForce, s= 150, color = 'none', marker='o',edgecolors='darkblue', alpha = 0.8, linewidths=3)
    ax1.plot(range(len(MaxForce)), MaxForce, '-', color = 'darkblue', linewidth=3.0, )
    ax1.plot(range(len(MaxForce)), [MFTh]*len(MaxForce), '--', color = 'black', linewidth=2.0, )
    # ax1.set_xlim(np.min(self._xkpt), np.max(self._xkpt))
    ax1.set_ylim(0, 2*MFTh)
    ax1.set_xlabel('Iteration step', fontsize = 25, fontweight = 'semibold')
    ax1.set_ylabel('Max Force', fontsize = 25, fontweight = 'semibold', color = 'darkblue')
    ax1.tick_params(axis = 'both', labelsize = 25, width = 2, length = 8, direction='in')

    ax1.spines['bottom'].set_linewidth(3)
    ax1.spines['left'].set_linewidth(3) 
    ax1.spines['top'].set_linewidth(3)
    ax1.spines['right'].set_linewidth(3)  

    ax2 = plt.subplot(222)
    ax2.scatter(range(len(RMSForce)), RMSForce, s= 150, color = 'none', marker='o',edgecolors='darkblue', alpha = 0.8, linewidths=3)
    ax2.plot(range(len(RMSForce)), RMSForce, '-', color = 'darkblue', linewidth=3.0, )
    ax2.plot(range(len(RMSForce)), [RFTh]*len(RMSForce), '--', color = 'black', linewidth=2.0, )
    # ax1.set_xlim(np.min(self._xkpt), np.max(self._xkpt))
    ax2.set_ylim(0, 2*RFTh)
    ax2.set_xlabel('Iteration step', fontsize = 25, fontweight = 'semibold')
    ax2.set_ylabel('RMS Force', fontsize = 25, fontweight = 'semibold', color = 'darkblue')
    ax2.tick_params(axis = 'both', labelsize = 25, width = 2, length = 8, direction='in')

    ax2.spines['bottom'].set_linewidth(3)
    ax2.spines['left'].set_linewidth(3) 
    ax2.spines['top'].set_linewidth(3)
    ax2.spines['right'].set_linewidth(3)  

    ax3 = plt.subplot(223)
    ax3.scatter(range(len(MaxDis)), MaxDis, s= 150, color = 'none', marker='o',edgecolors='darkblue', alpha = 0.8, linewidths=3)
    ax3.plot(range(len(MaxDis)), MaxDis, '-', color = 'darkblue', linewidth=3.0, )
    ax3.plot(range(len(MaxDis)), [MDTh]*len(MaxDis), '--', color = 'black', linewidth=2.0, )
    # ax1.set_xlim(np.min(self._xkpt), np.max(self._xkpt))
    ax3.set_ylim(0, 2*MDTh)
    ax3.set_xlabel('Iteration step', fontsize = 25, fontweight = 'semibold')
    ax3.set_ylabel('Max Displacement', fontsize = 25, fontweight = 'semibold', color = 'darkblue')
    ax3.tick_params(axis = 'both', labelsize = 25, width = 2, length = 8, direction='in')

    ax3.spines['bottom'].set_linewidth(3)
    ax3.spines['left'].set_linewidth(3) 
    ax3.spines['top'].set_linewidth(3)
    ax3.spines['right'].set_linewidth(3)  

    ax4 = plt.subplot(224)
    ax4.scatter(range(len(RMSDis)), RMSDis, s= 150, color = 'none', marker='o',edgecolors='darkblue', alpha = 0.8, linewidths=3)
    ax4.plot(range(len(RMSDis)), RMSDis, '-', color = 'darkblue', linewidth=3.0, )
    ax4.plot(range(len(RMSDis)), [RDTh]*len(RMSDis), '--', color = 'black', linewidth=2.0, )
    # ax1.set_xlim(np.min(self._xkpt), np.max(self._xkpt))
    ax4.set_ylim(0, 2*RDTh)
    ax4.set_xlabel('Iteration step', fontsize = 25, fontweight = 'semibold')
    ax4.set_ylabel('RMS Displacement', fontsize = 25, fontweight = 'semibold', color = 'darkblue')
    ax4.tick_params(axis = 'both', labelsize = 25, width = 2, length = 8, direction='in')

    ax4.spines['bottom'].set_linewidth(3)
    ax4.spines['left'].set_linewidth(3) 
    ax4.spines['top'].set_linewidth(3)
    ax4.spines['right'].set_linewidth(3)  

    ax1.grid(linestyle='--')
    ax2.grid(linestyle='--')
    ax3.grid(linestyle='--')
    ax4.grid(linestyle='--')

    plt.tight_layout()
    plt.savefig('Opt.jpg', dpi = 300)

def check_opt(log_file):

    if os.popen("grep 'Optimization completed.' " + log_file).read().split("\n")[0]:
    
        print ("Optimization Completed !")

    elif os.popen("grep 'Optimization stopped.' " + log_file).read().split("\n")[0]:

        print ("Optimization is not complete, please modify the parameters !") 
    

def ExactSCFItera(log_file):

    pass

def ExtactESPcharge(log_file,):

    natoms = int(os.popen("grep -n 'NAtoms' " + log_file).read().split()[2])    
    line_num = int(os.popen("grep -n 'ESP charges:' " + log_file).read().split('\n')[-2].split(':')[0])
    bk = os.popen("sed -n " + str(line_num + 2) + ',' + str(line_num + 1 + natoms) + 'p ' + log_file).read().split()
    _species = bk[1:][::3]
    _charges = [float(i) for i in bk[2:][::3]]
    
    return pd.DataFrame({
        'Specie': _species,
        "Charge": _charges
    })

if __name__ == '__main__':


    if task == 'opt':
        ExtactEnergyItera(log)
        ExactOptCriitera(log)
        check_opt(log)
    
    elif task == 'scf':
        
        pass
