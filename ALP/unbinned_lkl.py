#!/usr/bin/python3.8
from gammapy.utils.random import get_random_state
import numpy as np
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import chi2
from functools import partial



class DifferentialCounts:
    """Differential counts dN/dE in energy from a dataset

    Parameters
    ----------
    dataset : a dataset from SpectrumDataset or MapDataser

    """
    
    def __init__(self, dataset):
        if not hasattr(dataset, 'evaluators'):
            raise ValueError("Object dataset has no attribute evaluators")
        else:
            self.dataset              = dataset
            evaluator                 = list( self.dataset.evaluators.values() )[0]
            self.exposure_flux        = evaluator.apply_exposure(evaluator.compute_flux())
            self.exposure_flux_edisp  = evaluator.apply_edisp(self.exposure_flux)
            self.counts               = self.exposure_flux.data
            self.counts_edisp         = self.exposure_flux_edisp.data
            self.signal               = np.sum(self.counts)
            self.signal_edisp         = np.sum(self.counts_edisp)
    

    def get_coords(self,estimated=True):
        """Get energy and differential counts

        Parameters
        ----------
        estimated : Bool
            True if dN/dE in estimted energy,
            i.e. by incluidng edisp effects
            By default is True

        Returns
        -------
        tuple of `~numpy.narray`
            Center of the energy bins,
            Width of the energy bins ,
            The corresponding dN/dE
        """
        
        if estimated:
            axes        =   self.exposure_flux_edisp.geom.axes
            energy_axe  =   axes['energy']
            counts      =   self.counts_edisp
        else:
            axes        =   self.exposure_flux.geom.axes
            energy_axe  =   axes['energy_true']
            counts      =   self.counts

        while len(counts.shape) > 1:
            counts      = counts.sum(axis=1)

        bin_centers      = np.array(energy_axe.center)
        bin_width        = np.array(energy_axe.bin_width)
        
        return bin_centers, bin_width, counts

    def plot(self, ax=None, fig=None, estimated=True,line=True,**kwargs):
        """Plot the differential counts dN/dE

        Parameters
        ----------
        estimated : Bool
            True if dN/dE in estimted energy,
            i.e. by incluidng edisp effects
            By default is True

        """
        ax.grid(True,which='both',linewidth=0.8)
        ax.set_ylabel('dN / dE  [1/TeV]',size=30)
        ax.set_xlabel('E [TeV]',size=30)
        ax.set_xscale("log")
        ax.set_yscale("log")
        
        bin_centers, bin_width, counts = self.get_coords(estimated=estimated)
        dnde = counts/bin_width
        if line:
            ax.plot(bin_centers,dnde,**kwargs)
        else:
            ax.scatter(bin_centers,dnde,**kwargs)
        return ax, fig

    def get_dnde(self,energy,estimated=True):
        xp, bin_width , counts = self.get_coords(estimated=estimated)
        #counts                 = counts/ np.sum(counts)
        fp                     = counts/bin_width
        dnde                   = np.interp(energy, xp, fp, left=0, right=None)
        return dnde
    
    
    def simulate_energies(self, nevents=None, estimated=True, random_state="random-seed"):
        """Draw sample from the differental conts dN/dE.

        Parameters
        ----------
        size : int
            Number of samples to draw.
        estimated : Bool
            True if dN/dE in estimted energy,
            i.e. by incluidng edisp effects
            By default is True

        Returns
        -------
        en_array :  `~numpy.ndarray`
            Simulated energy events
        """
        
        if nevents is None:
            random_state      = get_random_state(random_state)
            if estimated:
                nevents       = random_state.poisson(self.signal_edisp)
            else:
                nevents       = random_state.poisson(self.signal)

        #coords    = self.exposure_flux.sample_coord(n_events=nevents, random_state=random_state)

        
        if estimated:
            #coords_reco  = self.dataset.edisp.sample_coord(coords, random_state=random_state)
            coords_reco  = self.exposure_flux_edisp.sample_coord(n_events=nevents, random_state=random_state)
            en_array     = coords_reco["energy"]
            en_array     = en_array [~np.isnan(en_array )]
        else:
            coords    = self.exposure_flux.sample_coord(n_events=nevents, random_state=random_state)
            en_array  = coords["energy_true"]

        return  en_array

def CDF_from_list(arr, x):
    if hasattr(x, "__len__"):
        x = [i for i in x]
    else:
        x = [x]
    freq      = np.histogram(arr,bins= [-np.inf] + x )[0]/arr.size
    cdf       = np.cumsum(freq)
    return cdf


class GridLikelihood:
    def __init__(self, model_per_gridpoint, log_likelihood_function,
                 TS_per_gridpoint=None, sim_TSs_per_gridpoint = None, CL_per_gridpoint=None,
                events_list =None):
        self.model_per_gridpoint     = model_per_gridpoint
        self.TS_per_gridpoint        = TS_per_gridpoint
        self.sim_TSs_per_gridpoint   = sim_TSs_per_gridpoint
        self.CL_per_gridpoint        = CL_per_gridpoint
        #
        self.log_likelihood_function = log_likelihood_function
        self.events_list             = events_list
    
    def get_CL_per_gridpoint(self,TS_cdf = None, number_of_simulations_per_point=None,
                            redo_simulations=False):
        
        if self.TS_per_gridpoint is None:
            self.get_TS_per_gridpoint()

        CL_per_gridpoint = {}
        for mg in self.model_per_gridpoint.keys():
            TSobs     = self.TS_per_gridpoint[mg]
        
            if number_of_simulations_per_point is None:
                if TS_cdf is None: #  we trust the Wilks theorem
                    CL_per_gridpoint[mg] = chi2.cdf(TSobs,df=2)
                else:
                    CL_per_gridpoint[mg] = TS_cdf(T_obs)
            else: # we are going to obtain the TS distribution for each gridpoint
                if self.sim_TSs_per_gridpoint is None:
                    self.get_sim_TSs_per_gridpoint(number_of_simulations_per_point)
                sim_TSs   = self.sim_TSs_per_gridpoint[mg]
                CL_per_gridpoint[mg] = CDF_from_list(sim_TSs, TSobs)

        self.CL_per_gridpoint = CL_per_gridpoint

    def get_sim_TSs_per_gridpoint(self,number_of_simulations_per_point=100,which_points=None):
        
        sim_TSs_per_gridpoint = {}
        if which_points is None:
            which_points = self.model_per_gridpoint.keys()
        
        totp = len(which_points)
        for k,mg in enumerate(which_points):
            print(str(k+1)+" / "+str(totp)+" ... ")
            print("We are simulating "+str(number_of_simulations_per_point)+
                  " TS values for gridpoint "+str(mg))
            dnde    = self.model_per_gridpoint[mg]
            TS_list = []
            for i in range(number_of_simulations_per_point):
                mc_events = dnde.simulate_energies(estimated=True)
                TS_values = self.get_TS_per_gridpoint(fake_events_list = mc_events)
                TS        = TS_values[mg]
                TS_list.append(TS)
            sim_TSs_per_gridpoint[mg] = np.array(TS_list)
            self.sim_TSs_per_gridpoint = sim_TSs_per_gridpoint


    def get_TS_per_gridpoint(self, fake_events_list = None):
        
        if fake_events_list is None and self.events_list is None:
            raise ValueError("No events list provided!")
    
        if fake_events_list is None:
            en_array   = self.events_list
        else:
            en_array   = fake_events_list


        logL_dict = {}
        TS_dict   = {}
        min_logL = None
        for mg in self.model_per_gridpoint.keys():
            dnde           = self.model_per_gridpoint[mg]
            TeVen_array    = en_array.to(u.TeV)
            logL           = -2*self.log_likelihood_function(TeVen_array.value,dnde)
            logL_dict[mg]  = logL
            TS_dict[mg]    = logL
            if min_logL is None:
                min_logL      = logL
            elif logL < min_logL:
                min_logL = logL
        # subtract the minimum to get the TS
        for i in TS_dict.keys(): TS_dict[i] -= min_logL

        if fake_events_list is None:
            self.min_logL            = min_logL
            self.logL_per_gridpoint  = logL_dict
            self.TS_per_gridpoint    = TS_dict
        else:
            return TS_dict

    def plot_cdf(self, fig,ax, which_points=None,legend=True):
        if which_points is None:
            which_points = self.model_per_gridpoint.keys()
        
        num_plots = len(which_points)
        #colormap = plt.cm.gist_ncar
        #plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, num_plots))))
        for k,mg in enumerate(which_points):
            TSvals = self.sim_TSs_per_gridpoint[mg]
            cdf    = partial(CDF_from_list, TSvals)
            x      = np.linspace(0,np.max(TSvals),1000)
            ax.plot(x, cdf(x), label="Gridpoint "+str(k))

        ax.plot(x, chi2.cdf(x,df=2),color="black",label="$\chi^2_{df=2}$")

        if legend:
            ncol = int(num_plots/15)
            ax.legend(ncol=ncol, fontsize=5,loc='center left', bbox_to_anchor=(1, 0.5),
            columnspacing=1.0, labelspacing=0.0,
            handletextpad=0.0, handlelength=1.5,
            fancybox=True, shadow=True)

        return fig, ax
    


    def plot_grid(self, fig,ax, what_to_plot=None, show_map=True,scatter=False,
                         true_val=None,show_values=False, show_lines=True,**kwargs):
        ''' plot likelihood reuslt for m and g

        '''
        
        if what_to_plot == "TS":
            zlabel = '$-2 \Delta \;  \log \; (\;  L \;)$'
            if self.TS_per_gridpoint is None:
                raise ValueError("No TS values for each grid point! Please run 'get_TS_per_gridpoint'")
            else:
                grid    = self.TS_per_gridpoint
        if what_to_plot == "logL":
            zlabel = '$-2 \;  \log \; (\;  L \;)$'
            if self.TS_per_gridpoint is None:
                raise ValueError("No logL values for each grid point! Please run 'get_TS_per_gridpoint'")
            else:
                grid     = self.logL_per_gridpoint
        if what_to_plot == "CL":
            zlabel = 'CL'
            if self.CL_per_gridpoint is None:
                raise ValueError("No CL values for each grid point! Please run 'get_CL_per_gridpoint'")
            else:
                grid     = self.CL_per_gridpoint


        Z      = np.array( list( grid.values() ) )
        Z      = Z.transpose()
        points = np.array(list(grid.items()), dtype=object)[:,0]
        x      = u.Quantity( [i[0] for i in points] )
        y      = u.Quantity( [i[1] for i in points] )

        if x.unit == u.eV:
            m = x
        elif x.unit == 1/u.GeV:
            g = x
        else:
            raise ValueError("Coordinates dimension should be eV for mass and 1/GeV for the coupling!")

        if y.unit == u.eV:
            m = y
        elif y.unit == 1/u.GeV:
            g = y
        else:
            raise ValueError("Coordinates dimension should be eV for mass and 1/GeV for the coupling!")

        X, Y  = np.meshgrid(np.unique(m.value), np.unique(g.value))
        Z     = np.reshape(Z, X.shape )

        # contour lines
        if show_lines:
            co    = ax.contour(  X, Y, Z, **kwargs)
            ax.clabel(co, inline=1, fontsize=20,fmt='%1.2f',colors='black')
        #
        # countor plot
        if show_map:
            co_f  = ax.contourf( X,Y , Z, 500, cmap="rainbow",vmax=np.max(Z))
            cbar  = fig.colorbar(co_f)
            cbar.set_label(zlabel, rotation=90,size=30)

        # optional things
        if show_values:
            for i, txt in enumerate(Z):
                for j, itxt in enumerate(txt):
                    itxt = round(itxt,2)
                    ax.annotate(itxt, (X[i][j], Y[i][j]),color="w",fontsize=20)
        if true_val is not None:
            ax.scatter(true_val[0],true_val[1],c="black",s=50,label="True value")
            ax.legend(loc='best')

        if scatter:
            ax.scatter(X,Y, s=50, facecolors='white', edgecolors='black',linewidths=2)

        return fig, ax



def compute_ALP_absorption(modulelist, axion_mass, coupling, emin, emax, bins):
    ''' Input:
            -  modulelist:     ModuleList object assuming a given source
            -  axion_mass:     axion mass / 1 neV
            -  coupling  :     axion-gamma coupling / 1e-11 GeV^-1
            -  emin      :     min energy / GeV
            -  emin      :     max energy / GeV
            -  bins      :     number of points in energy log-sperated
        Output:
            -  energy points
            -  gamma absorption for the above energy points

    '''
    ebins            = np.logspace(np.log10(emin),np.log10(emax),bins)
    
    # unit conversion
    axion_mass       = axion_mass.to(u.neV)
    coupling         = coupling.to(1e-11/u.GeV)
    # passing m and g to the gammaALPs modulelist object
    modulelist.alp.m = axion_mass.value
    modulelist.alp.g = coupling.value
    modulelist.EGeV  = ebins

    px,  py,  pa     = modulelist.run(multiprocess=2)
    pgg              = px + py
    
    return modulelist.EGeV, pgg

def plot_counts_in_energy(axes,energy_list, emin,emax,en_bins,**kwargs):
    ''' array of energies must be in TeV
    '''
    axes.grid(True,which='both',linewidth=0.8)
    axes.set_ylabel('dN / dE  [1/TeV]',size=30)
    axes.set_xlabel('E [TeV]',size=30)
    axes.set_xscale("log")
    axes.set_yscale("log")
    
    bins         = np.logspace(np.log10(emin),np.log10(emax),en_bins)
    y,bin_edges  = np.histogram(energy_list,bins=bins)
    binwidth     = bin_edges[1:]-bin_edges[:-1]
    bincenters   = np.sqrt(bin_edges[1:]*bin_edges[:-1])
    menStd       = np.sqrt(y)/binwidth
    y            = y/binwidth
    
    axes.errorbar(bincenters, y , menStd, fmt='o',**kwargs, elinewidth=2, markersize=5, capsize=4)
    
    return axes


def previous_limits(ax,plot_HESS=True,plot_Fermi=True,
                        plot_Mrk421=True, plot_Helioscopes=True,
                        plot_Chandra = True):
    if plot_HESS:
        HESS(ax)
    if plot_Mrk421:
        Mrk421(ax)
    if plot_Fermi:
        Fermi(ax)
    if plot_Chandra:
        Chandra(ax)
    if plot_Helioscopes:
        Helioscopes(ax)
    return ax

def HESS(ax,text_label=r'{\bf HESS}',text_pos=[3e-8,3e-11],col=[0.0, 0.55, 0.3],text_col="w",fs=16,zorder=0.2,text_on=True):
    # HESS arXiv:[1304.0700]
    dat = np.loadtxt("AxionPhoton/HESS.txt")
    FilledLimit(ax,dat,text_label,text_pos=text_pos,col=col,text_col=text_col,fs=fs,zorder=zorder,text_on=text_on)
    return


def Mrk421(ax,text_label=r'{\bf Mrk 421}',text_pos=[3e-9,6e-11],col=[0.4, 0.6, 0.1],text_col='w',fs=12,zorder=0.26,text_on=True):
    # Mrk 421 arXiv:[2008.09464]
    dat = np.loadtxt("AxionPhoton/Mrk421.txt")
    FilledLimit(ax,dat,text_label,text_pos=text_pos,col=col,text_col=text_col,fs=fs,zorder=zorder,text_on=text_on)
    return ax

def Chandra(ax,text_label=r'{\bf Chandra}',text_pos=[1.01e-11,1.5e-12],col= [0.0, 0.3, 0.24],text_col=[0.0, 0.3, 0.24],fs=15,zorder=0.1,text_on=True):
    # Chandra arXiv:[1907.05475]
    dat = np.loadtxt("AxionPhoton/Chandra.txt")
    FilledLimit(ax,dat,text_label,text_pos=text_pos,col=col,text_col=text_col,fs=fs,zorder=zorder,text_on=text_on)
    return

def Fermi(ax,text_label=r'{\bf Fermi}',text_pos=[4.02e-9,1.2e-11],col=[0.0, 0.42, 0.24],text_col='w',fs=15,zorder=0.24,text_on=True):
    # Fermi NGC1275 arXiv:[1603.06978]
    Fermi1 = np.loadtxt("AxionPhoton/Fermi1.txt")
    Fermi2 = np.loadtxt("AxionPhoton/Fermi2.txt")
    plt.fill_between(Fermi1[:,0],Fermi1[:,1],y2=1e0,edgecolor=col,facecolor=col,zorder=zorder,lw=3)
    plt.fill(Fermi2[:,0],1.01*Fermi2[:,1],edgecolor=col,facecolor=col,lw=3,zorder=zorder)
    Fermi1 = np.loadtxt("AxionPhoton/Fermi_bound.txt")
    Fermi2 = np.loadtxt("AxionPhoton/Fermi_hole.txt")
    plt.plot(Fermi1[:,0],Fermi1[:,1],'k-',alpha=0.5,lw=2,zorder=zorder)
    plt.plot(Fermi2[:,0],Fermi2[:,1],'k-',alpha=0.5,lw=2,zorder=zorder)
    if text_on:
        plt.text(text_pos[0],text_pos[1],text_label,fontsize=fs,color=text_col,ha='left',va='top',clip_on=True)
    return

def Helioscopes(ax,col=[0.5, 0.0, 0.13],fs=25,projection=False,RescaleByMass=False,text_on=True):
    # CAST arXiv:[1705.02290]
    y2 = ax.get_ylim()[1]
    if RescaleByMass:
        rs1 = 1.0
        rs2 = 0.0
    else:
        rs1 = 0.0
        rs2 = 1.0
    dat = np.loadtxt("AxionPhoton/CAST_highm.txt")
    plt.plot(dat[:,0],dat[:,1]/(rs1*2e-10*dat[:,0]+rs2),'k-',lw=2,zorder=1.49,alpha=1)
    plt.fill_between(dat[:,0],dat[:,1]/(rs1*2e-10*dat[:,0]+rs2),y2=y2,edgecolor='k',facecolor=col,zorder=1.49,lw=0.1)
    mf = dat[-2,0]
    gf = dat[-2,1]
    dat = np.loadtxt("AxionPhoton/CAST.txt")
    plt.plot(dat[:,0],dat[:,1]/(rs1*2e-10*dat[:,0]+rs2),'k-',lw=2,zorder=1.5,alpha=1)
    plt.fill_between(dat[:,0],dat[:,1]/(rs1*2e-10*dat[:,0]+rs2),y2=y2,edgecolor='none',facecolor=col,zorder=1.5,lw=0.0)
    gi = 10.0**np.interp(np.log10(mf),np.log10(dat[:,0]),np.log10(dat[:,1]))/(rs1*2e-10*mf+rs2)
    plt.plot([mf,mf],[gf,gi],'k-',lw=2,zorder=1.5)
    if text_on==True:
        if rs1==0:
            plt.text(4e-8,8.6e-11,r'{\bf CAST}',fontsize=fs+2,color='w',rotation=0,ha='center',va='top',clip_on=True)
        else:
            plt.text(4e-8,8e-11,r'{\bf CAST}',fontsize=fs+4,color='w',rotation=0,ha='center',va='top',clip_on=True)

    if projection:
        # IAXO arXiv[1212.4633]
        IAXO_col = 'purple'
        IAXO = np.loadtxt("AxionPhoton/Projections/IAXO.txt")
        plt.plot(IAXO[:,0],IAXO[:,1]/(rs1*2e-10*IAXO[:,0]+rs2),'--',linewidth=2.5,color=IAXO_col,zorder=0.5)
        plt.fill_between(IAXO[:,0],IAXO[:,1]/(rs1*2e-10*IAXO[:,0]+rs2),y2=y2,edgecolor=None,facecolor=IAXO_col,zorder=0,alpha=0.3)
        if text_on==True:
            if rs1==0:
                plt.text(0.35e-1,0.2e-11,r'{\bf IAXO}',fontsize=fs,color=IAXO_col,rotation=45,clip_on=True)
            else:
                plt.text(0.7e-2,0.12e1,r'{\bf IAXO}',fontsize=fs,color=IAXO_col,rotation=-18,clip_on=True)
    return

def FigSetup(xlab=r'$m_a$',ylab='$|g_{a\gamma}|$',\
                 xunit = 'neV', yunit = '$GeV$^{-1}$',
                 g_min = 1.0e-19,g_max = 1.0e-6,\
                 m_min = 1.0e-12,m_max = 1.0e7,\
                 lw=2.5,lfs=45,tfs=25,tickdir='out',\
                 Grid=False,Shape='Rectangular',\
                 mathpazo=False,TopAndRightTicks=False,\
                xtick_rotation=20.0,tick_pad=8,\
             FrequencyAxis=False,N_Hz=1,upper_xlabel=r"$\nu_a$ [Hz]",**freq_kwargs):

    xlab +=' [ '+xunit+' ] '
    ylab +=' [ '+yunit+' ] '
    #plt.rcParams['axes.linewidth'] = lw
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif',size=tfs)

    if mathpazo:
        mpl.rcParams['text.latex.preamble'] = [r'\usepackage{mathpazo}']

    if Shape=='Wide':
        fig = plt.figure(figsize=(16.5,5))
    elif Shape=='Rectangular':
        fig = plt.figure(figsize=(16.5,11))

    ax = fig.add_subplot(111)

    ax.set_xlabel(xlab,fontsize=lfs)
    ax.set_ylabel(ylab,fontsize=lfs)

    ax.tick_params(which='major',direction=tickdir,width=2.5,length=13,right=TopAndRightTicks,top=TopAndRightTicks,pad=tick_pad)
    ax.tick_params(which='minor',direction=tickdir,width=1,length=10,right=TopAndRightTicks,top=TopAndRightTicks)

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([m_min,m_max])
    ax.set_ylim([g_min,g_max])

    locmaj = mpl.ticker.LogLocator(base=10.0, subs=(1.0, ), numticks=50)
    locmin = mpl.ticker.LogLocator(base=10.0, subs=np.arange(2, 10)*.1,numticks=100)
    ax.xaxis.set_major_locator(locmaj)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    locmaj = mpl.ticker.LogLocator(base=10.0, subs=(1.0, ), numticks=100)
    locmin = mpl.ticker.LogLocator(base=10.0, subs=np.arange(2, 10)*.1,numticks=100)
    ax.yaxis.set_major_locator(locmaj)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    plt.xticks(rotation=xtick_rotation)

    if Grid:
        ax.grid(zorder=0)

    if FrequencyAxis:
        UpperFrequencyAxis(ax,N_Hz=N_Hz,tickdir='out',\
                           xtick_rotation=xtick_rotation,\
                           xlabel=upper_xlabel,\
                           lfs=lfs/1.3,tfs=tfs,tick_pad=tick_pad-2,**freq_kwargs)

    return fig,ax

def FilledLimit(ax,dat,text_label='',col='ForestGreen',edgecolor='k',zorder=1,\
                    lw=2,y2=1e0,edgealpha=0.6,text_on=False,text_pos=[0,0],\
                    ha='left',va='top',clip_on=True,fs=15,text_col='k',rotation=0,facealpha=1):
    plt.plot(dat[:,0],dat[:,1],'-',color=edgecolor,alpha=edgealpha,zorder=zorder,lw=lw)
    plt.fill_between(dat[:,0],dat[:,1],y2=y2,edgecolor=None,facecolor=col,alpha=facealpha,zorder=zorder)
    if text_on:
        plt.text(text_pos[0],text_pos[1],text_label,fontsize=fs,color=text_col,ha=ha,va=va,clip_on=clip_on,rotation=rotation,rotation_mode='anchor')
    return



