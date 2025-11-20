#!/usr/bin/python3
# Calculates intensity from star of given mass at given exoplanet semi-major axis
# Then calculates the photosynthetic rate at that location
# Calculates the habitable zone for aa given star using Kopparapu et al.(2014) paper.
# http://depts.washington.edu/naivpl/sites/default/files/hz_0.shtml#overlay-context=content/hz-calculator

# To-do - interpolate between the stellar massess so have more data points.

#import pyfits
import matplotlib
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt,numpy as np
import copy
import math
import scipy.interpolate
import pandas as pd


#Open the billion year old seiss model. Warning - I cut off all masses with Teff above 7200K because of
# the range given in Kopparupu 2014

print("Warning: if use interpolated values, the indices will be different because \
you delete spectral type")

#Using a file with fewer interpolations runs faster
isochrone_file = '5interpolated_seiss_1E9'
filename = isochrone_file + '.dat'

AU = 149597870700
AU_cgs = AU*100.0
PC = 3.08567758E16
pi = 3.141592654
Rsol = 6.957E8
Msol = 1.989E30
Msol_cgs = Msol * 1000.0
Lsol = 3.828E26
c_light = 3.0E8
h_planck = 6.63E-34
k_boltzmann=1.38E-23
avogadro = 6.02214086E23
micromole = 1.0E-6 * avogadro
mjup = 1.898E27
mearth = 5.972E24

#************************************************************************************
# Coeffcients to be used in the analytical expression to calculate habitable zone flux
# boundaries, Kopparapu et al. (2013) doi:10.1088/0004-637X/765/2/131

print("WARNING***********************************************************")
print("check the coefficients - there was an erratum printed, and the numbers do not match")


#The order is:
# Recent Venus, Runaway Greenhouse, Maximum Greenhouse, Early mars, Runaway Greenhouse for 5 ME,Runaway Greenhouse for 0.1 ME
seff = [0,0,0,0,0,0]

#These are the values for Sun system.
seffsun  = [1.776,1.107, 0.356, 0.320, 1.188, 0.99]
a = [2.136e-4, 1.332e-4, 6.171e-5, 5.547e-5, 1.433e-4, 1.209e-4]
b = [2.533e-8, 1.580e-8, 1.698e-9, 1.526e-9, 1.707e-8, 1.404e-8]
c = [-1.332e-11, -8.308e-12, -3.198e-12, -2.874e-12, -8.968e-12, -7.418e-12]
d = [-3.097e-15, -1.931e-15, -5.575e-16, -5.011e-16, -2.084e-15, -1.713e-15]

#************************************************************************************

#Minimum and maximum photosynthetically active wavelengths
lambda_max = 700.0E-9
lambda_min = 400.0E-9

#The step resolution to integrate over
nsteps = 400
dlambda = (lambda_max - lambda_min)/float(nsteps)

#=================================
# Values taken from model 3 of Yang et al. "quantifying photosynthetic performance of phytoplankton based
#on photosynthesis-irradiance response model"

alpha = 1.0E-5
beta = 0.001
gamma = 2

#=================================

#The exoplanet separation to consider
#This is in logspace, so 10^-2 = 0.01 au, 10^2.175 = 150 au, 10^3 = 1000 au etc

amin=0.001
amax=10.0

#number of samples for planet semi-major axis (needs to be ~10,000)
#nsample_aplanet = 1000

#print "*********************************************"

#print "Warning: change nsample_aplanet to be large (10,000 or so) for publication, or region not converged.\n MUST BE SQUARE"

#The array of planet separations
#a_planet = np.logspace(amin, amax,nsample_aplanet, base=10.0)


#Define stellar mass, stellar radius, and effective stellar temperature
mstar = []
rstar   = []
tstar   = []
lstar = []


# Define the photon rate
n_photon_rate_star =[]

#Read stellar mass, stellar radius, and effective stellar temperature from baraffe file
with open(filename) as f:
    for line in f:
        line     = line.strip()
        columns = line.split()

        mstar.append(float(columns[3]))
        tstar.append(float(columns[2]))

        #Temporary luminosity
        temp_L = float(columns[0])
        ##temp_L = 10.0**temp_L
        temp_L = temp_L * Lsol

        lstar.append(float(temp_L))
        #Only need to convert Rstar to solar radii

        rstar.append(float(columns[1]) * Rsol)
        
print(mstar)

#Make it a square distribution (maybe multiply by 20)
nsample_aplanet = len(mstar)*10

a_planet = np.linspace(amin, amax,nsample_aplanet)
#a_planet = np.logspace(amin, amax,nsample_aplanet, base=10.0)

#Put into SI units
a_planet = a_planet * AU

#For each star and planet separation, get S_eff
# Effective flux for that star (S_eff = S/S_0, where S = flux received at surface of planet, S_0 is solar flux received at earth)
S_0 = Lsol/ (4.* pi* AU**2)

# a is the indices in the equation of Koparupu 2014

S_eff_star=np.zeros((len(mstar),len(a)))

distance_line = np.zeros((len(mstar),len(a)))

#Now loop over every star, and every planet separation
for i in range(0,len(mstar)):
        for j in range(len(a)):
                #From Kopparapu equation 2, Tstar = T_effective - 5780K
                Tstar = tstar[i] - 5780

                seff[j] = seffsun[j] + a[j]*Tstar+ b[j]*Tstar**2 + c[j]*Tstar**3 + d[j]*Tstar**4
                S_eff_star[i,j] =  seff[j]


                distance_line[i,j] =  ((lstar[i]/Lsol) / S_eff_star[i,j])**0.5



recent_venus =  distance_line[:,0]
runaway_greenhouse = distance_line[:,1]
maximum_greenhouse =  distance_line[:,2]
early_mars =  distance_line[:,3]

runaway_greenhouse5Me = distance_line[:,4]

runaway_greenhouse01Me = distance_line[:,5]

# Recent Venus, Runaway Greenhouse, Maximum Greenhouse, Early mars, Runaway Greenhouse for 5 ME,Runaway Greenhouse for 0.1 ME

# plt.plot(runaway_greenhouse,mstar,c='r', label='Inner HZ')
# plt.plot(maximum_greenhouse,mstar,c='b',label='Outer HZ')
# plt.xlabel("Distance [au]")
# plt.ylabel(r"Host star mass [M$_\odot$]")
# # plt.xlim(0.02,5 )
# # plt.ylim(0.1,1.15)
# plt.loglog()
# plt.legend()
# plt.show()


# Now for each star, get the photon rate
for i in range(0,len(mstar)):

        lambda_last = lambda_min
        integral_sum = 0.0

        #Integrate over the number of steps for equation 3 of Lingam and Loeb 2020 https://arxiv.org/abs/1907.12576
        for j in range(0,nsteps):

                lambda_now = lambda_last + dlambda

                try:
                        integral = dlambda * (2.0 * c_light / lambda_now**4.)   * (( math.exp( (h_planck*c_light)
                                   / (lambda_now* k_boltzmann * tstar[i]))  -1.) **-1.)

                        #Note that we include a factor of pi unlike in Lingam and Loeb, since they missed out that factor, I think.
                        #http://spiff.rit.edu/classes/phys317/lectures/planck.html



                except:

                        integral = float('inf')

                #The new line I should be changing
                lambda_last= lambda_now

                integral_sum+=integral

        #Note that we include a factor of pi unlike in Lingam and Loeb, since they missed out that factor, I think.
        #http://spiff.rit.edu/classes/phys317/lectures/planck.html
        #Integral sum should be multiplied by pi here
        integral_sum = pi * integral_sum


        n_photon_rate = 4. * pi * rstar[i]**2. * integral_sum

        n_photon_rate_star.append(n_photon_rate)


#Allocate intensity received at the location of the planet
intensity_received  = np.zeros((len(mstar),nsample_aplanet))

#Allocate the synthetic rate from that intensity
photosynth_rate = np.zeros((len(mstar),nsample_aplanet))



#Allocate arrays to make a 2d grid
x=[]
y=[]
rate2D=[]

#NB (for now) these values give a 1 solar mass star at 1au, for sanity checking.
#i = 19, j = 400

#Want to draw some extra lines, so need to find the first and last place that the values are over and under a threshold.

#The threshold is just any region of parameter space for which the net photosynthesis rate will be positive.
threshold = 0.0

threshold_x_min = []
threshold_y_min = []
threshold_x_max = []
threshold_y_max = []



#Do we want to smooth these jaggy lines?
do_smoothing = True

if(do_smoothing):

        smooth_window = 51
        print("Warning: smoothing the plot - check results match non smoothed version")
        print("******************************************************************************************************")

        from scipy.ndimage.filters import gaussian_filter1d
        #threshold_y_min = gaussian_filter1d(threshold_y_min, sigma=5)
        #threshold_y_max = gaussian_filter1d(threshold_y_max, sigma=5)

        from scipy.signal import savgol_filter


        runaway_greenhouse = savgol_filter(runaway_greenhouse ,smooth_window,3)
        maximum_greenhouse = savgol_filter(maximum_greenhouse ,smooth_window,3)


        # recent_venus = gaussian_filter1d(recent_venus ,sigma=4)
        # runaway_greenhouse = gaussian_filter1d(runaway_greenhouse ,sigma=4)
        # maximum_greenhouse =  gaussian_filter1d(maximum_greenhouse ,sigma=4)
        # early_mars =  gaussian_filter1d(early_mars ,sigma=2)

        # runaway_greenhouse5Me = gaussian_filter1d(runaway_greenhouse5Me,sigma=4)

        # runaway_greenhouse01Me = gaussian_filter1d(runaway_greenhouse01Me,sigma=4)




#  threshold_y_max and min are different sizes, probably want to upsample to the same sized array.
# if(min_is_bigger):
#         threshold_x_min = padding(threshold_x_min, 0, len(threshold_x_max) )
# print np.shape(threshold_x_min)


# fig, ax = plt.subplots()
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
plt.rcParams['legend.numpoints'] = 1
# ax.loglog()
#ax.semilogy()


plt.xscale('log')
plt.yscale('log')

minx = 0.02
maxx = 10.0

miny = 0.1
maxy = 1.9

axes = 22
ticks = 18
legend = 18
title = 22

plt.xlim(minx,maxx)
plt.ylim(miny,maxy)

from matplotlib import rc,rcParams
# # activate latex text rendering
rc('text', usetex=True)





blue_fill = plt.scatter([], [], marker='s', color='dodgerblue', label='Habitable Zone',alpha=0.95,s=200)




#Label the plot with the respiration rate etc
#plt.text(0.03, 0.9, 'text', fontweight='bold',size=20)

#Deactivate the latex rendering after we have made the markers or it won't behave
rc('text', usetex=False)

# # # Fill
x1 = runaway_greenhouse.copy()
x2 = maximum_greenhouse.copy()
x3 = np.array(threshold_x_min)
x4 = np.array(threshold_x_max)

xfill = np.sort(np.concatenate([x1, x2, x3, x4]))
y1fill = np.interp(xfill, x1, mstar)
y2fill = np.interp(xfill, x2, mstar)



#crimson
photocolor = 'crimson'





# HZ
plt.fill_between(xfill, y1fill, y2fill, where=(y1fill > y2fill),
                  interpolate=True, color='dodgerblue', alpha=0.2)



makeplot = True

if(makeplot):

    #Do the actual plotting below here
    plotfont = 25

    #plt.title("The biosignature zone",  fontsize=plotfont,fontweight='bold')
    #plt.title("%s"%scenario,  fontsize=plotfont,fontweight='bold')
    plt.xlabel('Planet semi-major axis [au]', fontsize=plotfont,fontweight='bold')
    plt.ylabel(r'Host star mass [M$_\odot$]',  fontsize=plotfont,fontweight='bold')
    #plt.yticks(fontsize=ticks)
    #plt.xticks(fontsize=ticks)

    # Change legend fontsize
    matplotlib.rcParams['legend.fontsize'] = 20


    legend_properties = {'weight':'bold'}
    #plt.legend(loc='lower right',prop=legend_properties,framealpha=1.0)

    # These just plot nice thick lines on top of the existing axis
    ax.axhline(linewidth=8,y=maxy,color='k')
    ax.axhline(linewidth=8,y=miny,color='k')
    ax.axvline(linewidth=8,x=maxx,color='k')
    ax.axvline(linewidth=8,x=minx,color='k')

    from matplotlib import rc,rcParams
    # # activate latex text rendering
    # rc('text', usetex=True)
    rc('axes', linewidth=2)
    rc('font', weight='bold')
    plt.rcParams["axes.labelweight"] = "bold"
            #Make everything bold
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"


    #Set the tick labels to be bold here
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    plt.setp(ax.get_xticklabels(), fontweight="bold")

    # rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

    # Set everything for major and minor ticks
    ax.tick_params(direction='in',colors='k',length=24, width=4,which='major',labelcolor='k')
    ax.tick_params(direction='in',colors='k',length=10, width=4,which='minor',labelcolor='k')

    # Put the tick params on all axes, and pad the ticks away from the plot
    ax.tick_params(top=True,right=True,which='both',pad=15)

    ax.xaxis.set_tick_params(labelsize=plotfont)
    ax.yaxis.set_tick_params(labelsize=plotfont)



    plt.tight_layout()


    # order = [3, 4, 5, 0, 1, 2]
    # handles, labels = plt.gca().get_legend_handles_labels()
    # plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
    #            loc='lower right',legend
    #            fontsize=legend)


    savestring =  './' + isochrone_file + "_hz_only"
    filename= savestring + ".pdf"

    print("filename",filename)
    plt.savefig(filename,  bbox_inches='tight', format='pdf')


    outputname = savestring+'.eps'



    plt.show()
    plt.close()