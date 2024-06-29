#-*-coding:utf-8-*-
import os
import re
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod

class MolOrCoef(object):

    __metaclass__ = ABCMeta

    def __init__(self, gjf_file, log_file, ):
        
        
        self._gjf = gjf_file
        self._log = log_file

        __ = os.popen("sed -n '/^\s*$/=' " + self._gjf).read().split()    
        self._atom_num = int(__[2]) - int(__[1]) - 2
        self._atoms = np.array(os.popen("sed -n " + str(int(__[1])+2) + ',' + __[2] + 'p ' + self._gjf).read().split()).reshape(self._atom_num, -1)[:,0]
        self._coors = np.array(os.popen("sed -n " + str(int(__[1])+2) + ',' + __[2] + 'p ' + self._gjf).read().split()).reshape(self._atom_num, -1)[:,-3:]
        self._coors = self._coors.astype(np.float32)
        self._atom_index = {}
        for i_index, i in enumerate(self._atoms):
            self._atom_index[i+'-'+str(i_index+1)] = self._coors[i_index]
        
        final = int(os.popen("grep -n 'Molecular Orbital Coefficients' " + self._log).read().split()[-4][:-1])
        
        EigL = np.array(
                        os.popen("grep -n 'Eigenvalues -- ' " +  self._log + " | sed -n '/^\([0-9]\+\):.*$/p' | awk -v FS=':' -v start=" + str(final) + " '$1>start {print $1}'").read().split()
                        ).astype(int)
        
        self._el =  np.empty(shape = (0,), dtype=float)
        self._el_order = np.zeros(shape = (0,), dtype= float)  # energy level num
        self._state = []
        self._el_dict = {}
        self._basis_order = EigL[1]-EigL[0] - 3
        # print (EigL)
        for i_index, i in enumerate(EigL):
            if i <= EigL[-1]:
                info = os.popen("sed -n '" + str(i-2) + ',' + str(i) + "p' " + self._log).read().split('\n')[:-1]
                for j_index, j in enumerate(info):
                    if j_index == 0:
                        # print (info)
                        self._el_order = np.append(self._el_order, np.array(j.split()).astype(int), axis=0)
                    elif j_index == 1:
                        self._state += j.split()
                    elif j_index == 2:
                        self._el = np.concatenate((self._el, np.array(j.split()[2:]).astype(float)), axis = 0)

        self._el = self._el * 27.211

        for i_index, i in enumerate(self._state):
            self._el_dict[str(i_index+1) + '-' + i] = self._el[i_index]

        # self._all_coef = np.zeros(shape = (0, self._el_order.shape[0]), dtype=float)   # first dim should orbital number.
    
    @property
    def get_EnergyLevel(self):
        
        return self._el_dict

    @property
    def get_HomoLumo(self):
        
        return {"HOMO": self._el[self._state.index("V")-1],
                "LUMO": self._el[self._state.index("V")],}

    @abstractmethod
    def _ProCoefInfo(self):

        """
        'Basis set MO Coef' & 'NBO MO Coef'
        
        """

        pass

    @abstractmethod
    def ext_TolCoef(self):
        
        """
        | Energy |  Coef |  
        |  ** eV |  0.1  |
        ... 
        """
        pass

    @abstractmethod
    def ext_PartCoef(self,  *args, **kwargs):

        pass

    def plot_XDOS(self, type:str = 'TDOS', width = 4*np.sqrt(2*np.log(2)), fig_name = 'DOS', Orbital_setting:dict=None):
        """
        type = 'TDOS' or 'VDOS' or 'PDOS'

        For type is 'PDOS', 

        Please set Orbital type by dict data format,

        {
        'C':{'1,2':['s', 'p', 'dx2']},
        'S':{'7': ['tot']}
        }

        """
        
        import matplotlib.pyplot as plt
        from matplotlib.path import Path
        from matplotlib.patches import PathPatch
        from matplotlib.ticker import AutoMinorLocator
        from matplotlib.patches import Polygon
        import matplotlib.colors as mcolors
        
        if type == "TDOS":
            
            energy_range = np.arange(np.min(self._el), np.max(self._el), (np.max(self._el)-np.min(self._el))/100000)
            var = width /(2*np.sqrt(2*np.log(2)))
            GauExpansion = (1/(var*np.sqrt(2*np.pi))) * np.exp(-((np.expand_dims(energy_range, 0) - self._el.reshape(-1,1))**2/2*var**2))
            GauExpansion = np.sum(GauExpansion, axis=0)
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 7))
            ax1 = plt.subplot(111)
            line, = ax1.plot(energy_range, GauExpansion, '-', linewidth=2)
            
            z = np.empty((100, 1, 4), dtype=float)
            rgb = mcolors.colorConverter.to_rgb(line.get_color())
            z[:, :, :3] = rgb
            z[:, :, -1] = np.linspace(0.1, 1, 100).reshape(-1, 1)
            xmin, xmax, ymin, ymax = self._el.min(), self._el.max(), GauExpansion.min(), GauExpansion.max()
            im = ax1.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                            origin='lower',)
            xy = np.column_stack([energy_range, GauExpansion])
            xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
            clip_path = Polygon(xy, lw=0.0, facecolor='none',
                        edgecolor='none', closed=True)
            ax.add_patch(clip_path)
            im.set_clip_path(clip_path)
            ax1.axvline(x=self._el[self._state.index('V')-1], linestyle="--",color='b')
            ax1.axvline(x=self._el[self._state.index('V')], linestyle="--",color='r')
            for i in self._el:
                ax1.plot([i,i], [0,1], '-', c='black',linewidth=0.5)
            legend_font = {'weight': 'normal', "size" : 15}
            # ax1.legend(loc = 'upper right', ncol = 2, fontsize = 'large', prop = legend_font)
            ax1.set_xlim(-12, 2)
            ax1.set_ylim(0, GauExpansion.max() + 0.1*(GauExpansion.max()-GauExpansion.min()))
            # ax1.set_xticks([-4,-2,0,2,4,6,])
            # ax1.set_yticks([-2,0,2,4,6,])
            ax1.tick_params(axis = 'both', labelsize = 25, width = 2, length = 8, direction='in')
            ax1.set_xlabel('Energy', fontsize = 20, fontweight = 'medium')
            ax1.set_ylabel('Density of State', fontsize = 20, fontweight = 'medium')
            ax1.spines['bottom'].set_linewidth(2)
            ax1.spines['left'].set_linewidth(2)
            ax1.spines['top'].set_linewidth(2)
            ax1.spines['right'].set_linewidth(2)
            plt.tight_layout()
            plt.savefig('Tdos.jpg', dpi = 300)

        elif type == 'VDOS':

            val_NBO_MoOr_Coef_df = self._NBO_MoOr_Coef_df.T[self._NBO_MoOr_Coef_df.T.columns[self._ValSite]].T
            energy_range = np.arange(np.min(self._el), np.max(self._el), (np.max(self._el)-np.min(self._el))/100000)
            var = width /(2*np.sqrt(2*np.log(2)))
            GauExpansion = (1/(var*np.sqrt(2*np.pi))) * np.exp(-((np.expand_dims(energy_range, 0) - self._el.reshape(-1,1))**2/2*var**2))
            EL_p = []
            for i_index, i in enumerate(val_NBO_MoOr_Coef_df):
                p = (val_NBO_MoOr_Coef_df[[i]]**2).sum().to_numpy()[0]
                GauExpansion[i_index] = GauExpansion[i_index]*p
                EL_p += [p]
            GauExpansion = np.sum(GauExpansion, axis=0)

            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 7))
            ax1 = plt.subplot(111)
            line, = ax1.plot(energy_range, GauExpansion, '-', linewidth=2)
            z = np.empty((100, 1, 4), dtype=float)
            rgb = mcolors.colorConverter.to_rgb(line.get_color())
            z[:, :, :3] = rgb
            z[:, :, -1] = np.linspace(0.1, 1, 100).reshape(-1, 1)
            xmin, xmax, ymin, ymax = self._el.min(), self._el.max(), GauExpansion.min(), GauExpansion.max()
            im = ax1.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                            origin='lower',)
            xy = np.column_stack([energy_range, GauExpansion])
            xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
            clip_path = Polygon(xy, lw=0.0, facecolor='none',
                        edgecolor='none', closed=True)
            ax.add_patch(clip_path)
            im.set_clip_path(clip_path)
            ax1.axvline(x=self._el[self._state.index('V')-1], linestyle="--",color='b')
            ax1.axvline(x=self._el[self._state.index('V')], linestyle="--",color='r')
            for i_index, i in enumerate(self._el):
                ax1.plot([i,i], [0,1*EL_p[i_index]], '-', c='black',linewidth=0.5)
            legend_font = {'weight': 'normal', "size" : 15}
            # ax1.legend(loc = 'upper right', ncol = 2, fontsize = 'large', prop = legend_font)
            ax1.set_xlim(-20, 2)
            ax1.set_ylim(0, GauExpansion.max() + 0.1*(GauExpansion.max()-GauExpansion.min()))
            # ax1.set_xticks([-4,-2,0,2,4,6,])
            # ax1.set_yticks([-2,0,2,4,6,])
            ax1.tick_params(axis = 'both', labelsize = 25, width = 2, length = 8, direction='in')
            ax1.set_xlabel('Energy', fontsize = 20, fontweight = 'medium')
            ax1.set_ylabel('Density of State', fontsize = 20, fontweight = 'medium')
            ax1.spines['bottom'].set_linewidth(2)
            ax1.spines['left'].set_linewidth(2)
            ax1.spines['top'].set_linewidth(2)
            ax1.spines['right'].set_linewidth(2)
            plt.tight_layout()
            plt.savefig('Vdos.jpg', dpi = 300)
        
        elif type == 'PDOS':
            
            """
            {
            'C':{'1:3':['s', 'p', 'dx2']},  # s-p, tot, s-d, p-d 1-2 1,3,9  1:4
            'S':{'7': ['tot']}
            }

            """

            AtomOrType, OrbCom, EleOrder, AtomOrder = self._ProCoefInfo()
            val_NBO_MoOr_Coef_df = (self._NBO_MoOr_Coef_df.T[self._NBO_MoOr_Coef_df.T.columns[self._ValSite]].T)**2
            x = [i.split('-')[0] + '-' + i.split('-')[1] + '-' + i.split('-')[2][1:] for i in list(val_NBO_MoOr_Coef_df.T.columns)]
            Or_dict = {}
            Or_val_dict = {}
            Okeys = []
            
            for i_index, i in enumerate(Orbital_setting):
                if 'AtomSeq' not in i:

                    for j_index, j in enumerate(Orbital_setting[i]):
                        if ',' in j:
                            aor = j.split(',')
                            ac = [i + '-' + k for k in aor]
                        elif ':' in j:
                            aor = np.array(j.split(':')).astype(int)
                            aor = np.array(range(aor[0], aor[1] + 1))
                            ac = [i + '-' + str(k) for k in aor]
                        else:
                            ac = [i + '-' + str(j)]
                        for k_index, k in enumerate(Orbital_setting[i][j]):
                            if k == 'p':
                                acCom = [[m + '-px', m + '-py', m + '-pz',] for m in ac]
                            elif k == 'd':
                                acCom = [[m + '-dxy', m + '-dxz', m + '-dyz', m + '-dx2y2', m + '-dz2',] for m in ac]
                            else:
                                acCom = [m + '-' + k for m in ac]
                            Okeys += [acCom]
                        
                            if np.array(acCom).ndim == 1:
                                col = [x.index(xx) for xx in acCom]
                                Or_dict[i + '-' + j + '-' + k] = col
                                Or_val_dict[i + '-' + j + '-' + k] = np.sum(val_NBO_MoOr_Coef_df.T.to_numpy()[:, col], axis=1)
                            elif np.array(acCom).ndim == 2:
                                col = [[x.index(xx) for xx in xxx ] for xxx in acCom] 
                                cols = []
                                for m in col:
                                    cols+=m
                                Or_dict[i + '-' + j + '-' + k] = cols
                                Or_val_dict[i + '-' + j + '-' + k] = np.sum(val_NBO_MoOr_Coef_df.T.to_numpy()[:, cols], axis=1)
            
                else:

                    for j_index, j in enumerate(Orbital_setting[i]):
                        aor = j.split(',')
                        ac = []
                    
                        for k in aor:
                            if ":" not in k:
                                ac+=[EleOrder[AtomOrder.index(k)] + '-' + k]
                            else:
                                ar = list(range(int(k.split(':')[0]), int(k.split(':')[1]))) + [k.split(':')[1]]
                                ar = [str(m) for m in ar]
                                ac += [EleOrder[AtomOrder.index(m)] + '-' + m for m in ar]
                        
                        for k_index, k in enumerate(Orbital_setting[i][j]):
                            if k == 'p':
                                acCom = [[m + '-px', m + '-py', m + '-pz',] for m in ac]
                            elif k == 'd':
                                acCom = [[m + '-dxy', m + '-dxz', m + '-dyz', m + '-dx2y2', m + '-dz2',] for m in ac]
                            else:
                                acCom = [m + '-' + k for m in ac]
                            Okeys += [acCom]

                            if np.array(acCom).ndim == 1:
                                col = [x.index(xx) for xx in acCom]
                                Or_dict[i + '-' + j + '-' + k] = col
                                Or_val_dict[i + '-' + j + '-' + k] = np.sum(val_NBO_MoOr_Coef_df.T.to_numpy()[:, col], axis=1)
                            elif np.array(acCom).ndim == 2:
                                col = [[x.index(xx) for xx in xxx ] for xxx in acCom] 
                                cols = []
                                for m in col:
                                    cols+=m
                                Or_dict[i + '-' + j + '-' + k] = cols
                                Or_val_dict[i + '-' + j + '-' + k] = np.sum(val_NBO_MoOr_Coef_df.T.to_numpy()[:, cols], axis=1)
                    
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 7))
            ax1 = plt.subplot(111)
            for i_index, i in enumerate(Or_val_dict):
                
                energy_range = np.arange(np.min(self._el), np.max(self._el), (np.max(self._el)-np.min(self._el))/100000)
                var = width /(2*np.sqrt(2*np.log(2)))
                GauExpansion = (1/(var*np.sqrt(2*np.pi))) * Or_val_dict[i].reshape(len(self._el), -1) * (np.exp(-
                                                                                                                ((np.expand_dims(energy_range, 0) - self._el.reshape(-1,1))**2/2*var**2)
                                                                                                                ))
                GauExpansion = np.sum(GauExpansion, axis=0)
                line, = ax1.plot(energy_range, GauExpansion, '-', linewidth=2, label = i)
                z = np.empty((100, 1, 4), dtype=float)
                rgb = mcolors.colorConverter.to_rgb(line.get_color())
                z[:, :, :3] = rgb
                z[:, :, -1] = np.linspace(0.1, 1, 100).reshape(-1, 1)
                xmin, xmax, ymin, ymax = self._el.min(), self._el.max(), GauExpansion.min(), GauExpansion.max()
                im = ax1.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                               origin='lower',)
                xy = np.column_stack([energy_range, GauExpansion])
                xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
                clip_path = Polygon(xy, lw=0.0, facecolor='none',
                                    edgecolor='none', closed=True)     
                ax.add_patch(clip_path)
                im.set_clip_path(clip_path)
                for j_index, j in enumerate(self._el):
                    # print (j_index, ":" , Or_val_dict[i][j_index])
                    ax1.plot([j,j], [0, 1*Or_val_dict[i][j_index]], '-', c='black',linewidth=0.5)
    
            ax1.axvline(x=self._el[self._state.index('V')-1], linestyle="--",color='b')
            ax1.axvline(x=self._el[self._state.index('V')], linestyle="--",color='r')
            
            legend_font = {'weight': 'normal', "size" : 15}
            ax1.legend(loc = 'upper right', ncol = 2, fontsize = 'large', prop = legend_font)
            ax1.set_xlim(-20, 2)
            ax1.set_ylim(0, 1.2)
            # ax1.set_xticks([-4,-2,0,2,4,6,])
            # ax1.set_yticks([-2,0,2,4,6,])
            ax1.tick_params(axis = 'both', labelsize = 25, width = 2, length = 8, direction='in')
            ax1.set_xlabel('Energy', fontsize = 20, fontweight = 'medium')
            ax1.set_ylabel('Density of State', fontsize = 20, fontweight = 'medium')
            ax1.spines['bottom'].set_linewidth(2)
            ax1.spines['left'].set_linewidth(2)
            ax1.spines['top'].set_linewidth(2)
            ax1.spines['right'].set_linewidth(2)
            ax1.legend(ncol=1)
            plt.tight_layout()
            plt.savefig(fig_name +'.jpg', dpi = 300)


class NBOMolOrCoef(MolOrCoef):

    def __init__(self, gjf_file, log_file):

        super(NBOMolOrCoef, self).__init__(gjf_file, log_file)

    def _ProCoefInfo(self):
        
        try:
            nboL = int(os.popen("grep -n 'MOs in the NAO basis:' " + self._log).read().split("\n")[-2].split(":")[0])
        except FileExistsError:
            print ("Gaussian .gjf dont set the output of NBO molecular orbital parameters.")

        NBOMOLnum = np.array(os.popen("grep -n 'NAO' " + self._log + "| sed -n '/^\([0-9]\+\):.*$/p' | awk -v FS=':' -v start=" + str(nboL) + " '$1>start {print $1}'").read().split("\n")[:-1]).astype(int)
        self._NboMoOrNum = NBOMOLnum[1] - NBOMOLnum[0] - 3
        
        NAOOrL = int(os.popen("grep -n 'NATURAL POPULATIONS:  Natural atomic orbital occupancies' " + self._log).read().split('\n')[-2].split(':')[0])
        L = int(os.popen("grep -n 'Summary of Natural Population Analysis:' " + self._log).read().split('\n')[-2].split(':')[0])
        self.__AtomOrinfo = os.popen("sed -n '" + str(NAOOrL+4) + "," + str(L) + "p' " + self._log + " | grep '^[[:space:]].*[0-9][[:space:]]' | grep '[0-9].[0-9]\{5\}$'").read().split('\n')[:-1]
        AtomOrType = [i.split('(')[1].split(')')[0].strip().lower() for i in self.__AtomOrinfo]
        OrbCom = [i.split()[3].lower() for i in self.__AtomOrinfo]
        EleOrder = [i.split()[1] for i in self.__AtomOrinfo]
        AtomOrder = [i.split()[2] for i in self.__AtomOrinfo]
        
        self._NBO_MoOr_Coef = np.zeros(shape=(self._NboMoOrNum, 0))   # (Orbital num, Energy level num)
        for i_index, i in enumerate(NBOMOLnum):
            bp = os.popen("sed -n '" + str(i+2) + "," + str(i+1+self._NboMoOrNum) + "p'" + " " + self._log).read().split("\n")[:-1]
            bp = np.array(bp)
            ELcoef = np.array([i[17:].split() for i in bp]).astype(float)
            self._NBO_MoOr_Coef = np.concatenate((self._NBO_MoOr_Coef, ELcoef), axis = 1) 
        
        self._NBO_MoOr_Coef_df = pd.DataFrame(self._NBO_MoOr_Coef)
        self._NBO_MoOr_Coef_df.index =  [EleOrder[i] + '-' + str(AtomOrder[i]) + '-' + AtomOrType[i][:1] + OrbCom[i] for i in range(len(AtomOrType))]
        self._NBO_MoOr_Coef_df.columns = self._el_dict.keys()
        return AtomOrType, OrbCom, EleOrder, AtomOrder
    
    def IPR(self, if_plot = False):

        AtomOrType, OrbCom, EleOrder, AtomOrder = self._ProCoefInfo()
        __OrIndex = []
        __cache = []
        for i_index, i in enumerate(AtomOrder):
            if i not in __cache:
                __OrIndex.append(i_index)
                __cache.append(i)

        atomOrCoef_dict = {}
        for i_index, i in enumerate(self._NBO_MoOr_Coef_df.columns):
            atomOrCoef_dict[i] = []
            for j_index, j in enumerate(__OrIndex):
                if j_index != len(__OrIndex) - 1:
                    atomOrCoef_dict[i].append(self._NBO_MoOr_Coef_df[i][j:__OrIndex[j_index+1]])             
                else:
                    atomOrCoef_dict[i].append(self._NBO_MoOr_Coef_df[i][j:])

        proinfo = np.zeros(shape = (0, int(AtomOrder[-1])), dtype=float)
        for i_index, i in enumerate(atomOrCoef_dict):
            component = []
            for j_index, j in enumerate(atomOrCoef_dict[i]):
                component += [np.sum(np.square(j))]
            
            proinfo = np.append(proinfo, np.array([component]), axis=0)
        
        ipr = []
        for i_index, i in enumerate(proinfo):
            ipr+=[np.sum(np.square(i))/np.square(np.sum(i))]
        
        self._ipr = ipr
        self._NBO_MoOr_Coef_df.to_csv('nbo_coef.csv')
        np.square(self._NBO_MoOr_Coef_df).to_csv("nbo_percentage.csv")
        ipr_df = pd.DataFrame(ipr)
        ipr_df.index = self._el_dict.keys()
        ipr_df.columns = ['IPR']
        ipr_df.to_csv('IPR.csv')

        if if_plot:
            
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 7))
            ax1 = plt.subplot(111)
            for i_index, i in enumerate(self._ipr):
                
                ax1.plot([self._el[i_index], self._el[i_index]], [0, 1*self._ipr[i_index]], '-', c='black',linewidth=0.5)
            
            ax1.axvline(x=self._el[self._state.index('V')-1], linestyle="--",color='b')
            ax1.axvline(x=self._el[self._state.index('V')], linestyle="--",color='r')
            
            legend_font = {'weight': 'normal', "size" : 15}
            # ax1.legend(loc = 'upper right', ncol = 2, fontsize = 'large', prop = legend_font)
            ax1.set_xlim(-20, 2)
            ax1.set_ylim(0, 1)
            ax1.tick_params(axis = 'both', labelsize = 25, width = 2, length = 8, direction='in')
            ax1.set_xlabel('Energy', fontsize = 20, fontweight = 'medium')
            ax1.set_ylabel('Inverse participation ratio', fontsize = 20, fontweight = 'medium')
            ax1.spines['bottom'].set_linewidth(2)
            ax1.spines['left'].set_linewidth(2)
            ax1.spines['top'].set_linewidth(2)
            ax1.spines['right'].set_linewidth(2)
            plt.tight_layout()
            plt.savefig('IPR.jpg', dpi = 300)


        return self 
    
    def proipr(self, Frag_Orbital_setting: dict=None, if_plot = False, plot_name = 'proipr'):

        """
        Frag_Orbital_setting : {
        'C':{'1,2':['s', 'p', 'dx2']},
        'S':{'7': ['tot']}
        }

        """
        AtomOrType, OrbCom, EleOrder, AtomOrder = self._ProCoefInfo()
        val_NBO_MoOr_Coef_df = (self._NBO_MoOr_Coef_df.T[self._NBO_MoOr_Coef_df.T.columns[self._ValSite]].T)**2
        x = [i.split('-')[0] + '-' + i.split('-')[1] + '-' + i.split('-')[2][1:] for i in list(val_NBO_MoOr_Coef_df.T.columns)]
        Or_dict = {}
        Or_val_dict = {}
        Okeys = []

        for i_index, i in enumerate(Frag_Orbital_setting):
            if 'AtomSeq' not in i:
                for j_index, j in enumerate(Frag_Orbital_setting[i]):
                    if ',' in j:
                        aor = j.split(',')
                        ac = [i + '-' + k for k in aor]
                    elif ':' in j:
                        aor = np.array(j.split(':')).astype(int)
                        aor = np.array(range(aor[0], aor[1] + 1))
                        ac = [i + '-' + str(k) for k in aor]
                    else:
                        ac = [i + '-' + str(j)]
                    for k_index, k in enumerate(Frag_Orbital_setting[i][j]):
                        if k == 'p':
                            acCom = [[m + '-px', m + '-py', m + '-pz',] for m in ac]
                        elif k == 'd':
                            acCom = [[m + '-dxy', m + '-dxz', m + '-dyz', m + '-dx2y2', m + '-dz2',] for m in ac]
                        else:
                            acCom = [m + '-' + k for m in ac]
                        Okeys += [acCom]
                        if np.array(acCom).ndim == 1:
                            col = [x.index(xx) for xx in acCom]
                            Or_dict[i + '-' + j + '-' + k] = col
                            Or_val_dict[i + '-' + j + '-' + k] = val_NBO_MoOr_Coef_df.T.to_numpy()[:, col]
                        elif np.array(acCom).ndim == 2:
                            col = [[x.index(xx) for xx in xxx ] for xxx in acCom] 
                            cols = []
                            for m in col:
                                cols+=m
                            Or_dict[i + '-' + j + '-' + k] = cols
                            Or_val_dict[i + '-' + j + '-' + k] = val_NBO_MoOr_Coef_df.T.to_numpy()[:, cols]
            else:

                for j_index, j in enumerate(Frag_Orbital_setting[i]):
                    aor = j.split(',')
                    ac = []
                    
                    for k in aor:
                        if ":" not in k:
                            ac+=[EleOrder[AtomOrder.index(k)] + '-' + k]
                        else:
                            ar = list(range(int(k.split(':')[0]), int(k.split(':')[1]))) + [k.split(':')[1]]
                            ar = [str(m) for m in ar]
                            ac += [EleOrder[AtomOrder.index(m)] + '-' + m for m in ar]
                        
                    for k_index, k in enumerate(Frag_Orbital_setting[i][j]):
                        if k == 'p':
                            acCom = [[m + '-px', m + '-py', m + '-pz',] for m in ac]
                        elif k == 'd':
                            acCom = [[m + '-dxy', m + '-dxz', m + '-dyz', m + '-dx2y2', m + '-dz2',] for m in ac]
                        else:
                            acCom = [m + '-' + k for m in ac]
                        Okeys += [acCom]

                        if np.array(acCom).ndim == 1:
                            col = [x.index(xx) for xx in acCom]
                            Or_dict[i + '-' + j + '-' + k] = col
                            
                            Or_val_dict[i + '-' + j + '-' + k] = val_NBO_MoOr_Coef_df.T.to_numpy()[:, col]
                        elif np.array(acCom).ndim == 2:
                            col = [[x.index(xx) for xx in xxx ] for xxx in acCom] 
                            cols = []
                            for m in col:
                                cols+=m
                            Or_dict[i + '-' + j + '-' + k] = cols
                            
                            Or_val_dict[i + '-' + j + '-' + k] = val_NBO_MoOr_Coef_df.T.to_numpy()[:, cols]
                    
        project_ipr = []
        for i in Or_val_dict:
            bk = []
            for j_index, j in enumerate(Or_val_dict[i]):
                bk+= [np.sum(np.square(j))/np.square(np.sum(j))]
            project_ipr.append(bk)

        proipr_df = pd.DataFrame(np.array(project_ipr).T)
        
        proipr_df.index = self._el_dict.keys()
        proipr_df.columns = Frag_Orbital_setting.keys()
        proipr_df.to_csv('Project_IPR.csv')

        if if_plot:

            __color = ['royalblue', 'blueviolet', 'deeppink', 'mediumseagreen', 'mediumturquoise', 'dodgerblue', 'tomato', 'lightsea']
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 7))
            ax1 = plt.subplot(111)

            for i_index, i in enumerate(project_ipr):
                for j_index, j in enumerate(i):
                    if j_index == 0:
                        ax1.plot([self._el[j_index], self._el[j_index]], [0, 1*project_ipr[i_index][j_index]], '-',linewidth=0.8, c = __color[i_index], label = list(Or_val_dict.keys())[i_index])
                    else:
                        ax1.plot([self._el[j_index], self._el[j_index]], [0, 1*project_ipr[i_index][j_index]], '-',linewidth=0.8, c = __color[i_index],)

            ax1.axvline(x=self._el[self._state.index('V')-1], linestyle="--",color='b')
            ax1.axvline(x=self._el[self._state.index('V')], linestyle="--",color='r')
            
            legend_font = {'weight': 'normal', "size" : 15}
            ax1.legend(loc = 'upper right', ncol = 2, fontsize = 'large', prop = legend_font)
            ax1.set_xlim(-20, 2)
            
            ax1.tick_params(axis = 'both', labelsize = 25, width = 2, length = 8, direction='in')
            ax1.set_xlabel('Energy', fontsize = 20, fontweight = 'medium')
            ax1.set_ylabel('Inverse participation ratio', fontsize = 20, fontweight = 'medium')
            ax1.spines['bottom'].set_linewidth(2)
            ax1.spines['left'].set_linewidth(2)
            ax1.spines['top'].set_linewidth(2)
            ax1.spines['right'].set_linewidth(2)
            ax1.legend(ncol=1)
            plt.tight_layout()
            plt.savefig(plot_name + '.jpg', dpi = 300)
        
    
    def pro_OrbitalInfo(self,):
        
        AtomOrType, OrbCom, EleOrder, AtomOrder = self._ProCoefInfo()
        EleType = [i.split()[4].split('(')[0] for i in self.__AtomOrinfo]
        
        self._CorSite = [i_index for i_index, i in enumerate(EleType) if i=='Cor']
        self._ValSite = [i_index for i_index, i in enumerate(EleType) if i=='Val']
        self._RydSite = [i_index for i_index, i in enumerate(EleType) if i=='Ryd']
        EAO = np.array([EleOrder[i] + '-' + str(AtomOrder[i]) + '-' + AtomOrType[i][:1] + OrbCom[i] for i in range(len(AtomOrType))])
        
        return self

    def ext_AllMoCoef(self):
        
        return self._NBO_MoOr_Coef_df
    
    def ext_PartialMoCoef(self, ):
        """
        dict = {
        'Element 1': {'atom order 1':  ['s', 'px', 'd']}
        }

        """
        pass

class BasisMolOrCoef(MolOrCoef):

    def __init__(self, gjf_file, log_file):

        super(BasisMolOrCoef, self).__init__(gjf_file, log_file)
    
    def _ProCoefInfo(self):

        final = int(os.popen("grep -n 'Molecular Orbital Coefficients' " + self._log).read().split()[-4][:-1])
        # kws is the start line of orbital line -1  
        kws = np.array(
            os.popen("grep -n 'Eigenvalues --' " +  self._log + " | sed -n '/^\([0-9]\+\):.*$/p' | awk -v FS=':' -v start=" + str(final) + " '$1>start {print $1}'").read().split()
                                         ).astype(int)
        total_orbital = np.diff(kws)[0] - 3

        kws_b = kws[-1]
        
        self._Basis_MoOr_coef = np.zeros(shape = (0, total_orbital), dtype=float)
        self._Basis_MoOr_temp = []

        for i_index, i in enumerate(kws):
            if i <= kws_b:
                col_num = len(os.popen("sed -n '" + str(i-1) + "'p " + self._log).read().split())
                _or = os.popen("sed -n " + str(i+1) + "," + str(i + total_orbital) + "p " + self._log).read().split('\n')    
                _or.remove('')
                coef_temp = np.zeros(shape=(0, col_num), dtype=float)
                for j_index, j in enumerate(_or):
                    if i_index == 0:
                        coef_temp = np.append(coef_temp, np.array([re.findall(r'-?\d+\.\d+', j)]).astype(float), axis = 0)
                        other = j.split(re.findall(r'-?\d+\.\d+',j)[0])[0].split()
                        if len(other) != 2 and len(other) != 3:
                            ai = other[1]
                            el = other[2]
                        if other[-1] == '0':
                            self._Basis_MoOr_temp +=  [el  + '-' + ai + '-' + other[-2] + '-0']
                        else:
                            self._Basis_MoOr_temp += [el  + '-' + ai + '-' + other[-1]]
                        
                    else:
                        coef_temp = np.append(coef_temp, np.array([re.findall(r'-?\d+\.\d+', j)]).astype(float), axis = 0)
                self._Basis_MoOr_coef = np.concatenate((self._Basis_MoOr_coef, coef_temp.T), axis = 0)   # shape = (energy level number, orbital)

        self._Basis_MoOr_Coef_df = pd.DataFrame(self._Basis_MoOr_coef.T)
        self._Basis_MoOr_Coef_df.index = self._Basis_MoOr_temp
        self._Basis_MoOr_Coef_df.columns = self._el_dict.keys()
        
        return self

    def IPR(self):
        """
        Basis set coef is not useful for calculating IPR
        """
        AtomOrder = [i.split('-')[1] for i in list(self._Basis_MoOr_Coef_df.index)]
        __OrIndex = []
        __cache = []
        for i_index, i in enumerate(AtomOrder):
            if i not in __cache:
                __OrIndex.append(i_index)
                __cache.append(i)
    
        atomOrCoef_dict = {}
        for i_index, i in enumerate(self._Basis_MoOr_Coef_df.columns):
            atomOrCoef_dict[i] = []
            for j_index, j in enumerate(__OrIndex):
                if j_index != len(__OrIndex) - 1:
                    atomOrCoef_dict[i].append(self._Basis_MoOr_Coef_df[i][j:__OrIndex[j_index+1]])             
                else:
                    atomOrCoef_dict[i].append(self._Basis_MoOr_Coef_df[i][j:])

        proinfo = np.zeros(shape = (0, int(AtomOrder[-1])), dtype=float)
        for i_index, i in enumerate(atomOrCoef_dict):
            component = []
            for j_index, j in enumerate(atomOrCoef_dict[i]):
                component += [np.sum(np.square(j))]
            
            proinfo = np.append(proinfo, np.array([component]), axis=0)
       
        ipr = []
        for i_index, i in enumerate(proinfo):
           
            ipr+=[np.sum(np.square(i))/np.sum(i)]
        
        self._ipr = ipr
        return self 


if __name__ == "__main__":

    # Basis set Molecular orbital
    # bm = BasisMolOrCoef(gjf_file='PEDOT.gjf', log_file='gau.log')._ProCoefInfo().IPR()

    # For NBO local Orbital
    # MolOrCoef(gjf_file='C3H8.gjf', log_file='gau.log')
    nm = NBOMolOrCoef(gjf_file='mol.gjf', log_file='gau.log')
    
    nm.pro_OrbitalInfo().plot_XDOS(type='VDOS',
       )
    
    # nm.pro_OrbitalInfo().plot_XDOS(type='PDOS', Orbital_setting={
    #     'C':{'7,8,17,18,20,21,30,31,33,34,43,44,46,47,56,57':['pz']},
    #     'S':{"6,58":['pz'], "19,32,45":['pz']},
    #     'O':{"3,9,16,22,29,35,42,48,55,61":['pz']}}, 
    #     fig_name='pzSDos')
    
    nm.pro_OrbitalInfo().plot_XDOS(type='PDOS', Orbital_setting={
        "AtomSeq-1":{"4:8,17:21,30:34,43:47,56:60,3,9,16,22,29,35,42,48,55,61":['pz']},
        'C':{'4,5,7,8,17,18,20,21,30,31,33,34,43,44,46,47,56,57,59,60':['pz']},
        'S':{"6,58,19,32,45":['pz']},
        'O':{"3,9,16,22,29,35,42,48,55,61":['pz']}
        }, 
        fig_name='frag-pz-dos')
    
    nm.pro_OrbitalInfo().plot_XDOS(type='PDOS', Orbital_setting={
        "AtomSeq-1":{"4:8,17:21,30:34,43:47,56:60,3,9,16,22,29,35,42,48,55,61":['p']},
        'C':{'4,5,7,8,17,18,20,21,30,31,33,34,43,44,46,47,56,57,59,60':['p']},
        'S':{"6,58,19,32,45":['p']},
        'O':{"3,9,16,22,29,35,42,48,55,61":['p']}
        }, 
        fig_name='frag-p-dos')
    
    nm.pro_OrbitalInfo().proipr(
        Frag_Orbital_setting={
            "AtomSeq-1":{"4:8,17:21,30:34,43:47,56:60,3,9,16,22,29,35,42,48,55,61":['pz']}
            }, 
            if_plot=True,
            plot_name='frag-pz-ipr'
    )

    nm.pro_OrbitalInfo().proipr(
        Frag_Orbital_setting={
            "AtomSeq-1":{"4:8,17:21,30:34,43:47,56:60,3,9,16,22,29,35,42,48,55,61":['p']}
            }, 
            if_plot=True,
            plot_name='frag-p-ipr'
    )

    nm.pro_OrbitalInfo().IPR(
            if_plot=True
    )
