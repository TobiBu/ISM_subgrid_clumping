import yt
from astropy import units as u
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import itertools
import random
from yt.utilities.math_utils import get_cyl_theta, get_cyl_theta_component, euclidean_dist
import pickle
import re

import pandas as pd
import h5py

from numpy import tanh

from scipy.optimize import curve_fit
from scipy import stats


age = [0,1,2,3,4,5,6] #Myrs
plt.rc('axes', labelsize=18)

def mollw_proj(path, size_sphere, c, nside, prop = "cell_mass", do_proj = False):
    '''
    Extracts data and creates a Mollweide plot from an hdf5 dataset.

    Parameters
    -----------
    path       : the path to the simulation/snapshot file to be opened
    size_sphere: the radius of the sphere within the domain to be analyzed
    c          : coordinate of the center of the sphere
    nside      : parameter which determines the resolution of the map, usually a power of 2
    prop       : property to be projected
    do_proj    : if set to True, shows the Mollweide plot; set to False by default
    
    Returns
    --------
    An array of the quantity we are interested in. Optionally, Mollview image created using healpy.
    
    
    '''

    pc = 3.085e18
    ds = yt.load(path)

    center = c
    radius = (size_sphere, "pc")
    


    sphere = ds.sphere(center, radius)

    #quantity to be projected, default is cell mass, can be changed later
    quant = sphere[prop].v

    x_pos = sphere['x'].v
    y_pos = sphere['y'].v
    z_pos = sphere['z'].v
    
    left = [sphere['x'].min().v, sphere['y'].min().v, sphere['z'].min().v]

    n = sphere['x'].size
    #print("Number of particles is ", n)


    posit = np.dstack((x_pos, y_pos, z_pos)).reshape(n,3)


    dist1 = [euclidean_dist(center, posit[i]) for i in range(n)]
    pos1 = [[x_pos[i]-center[0],y_pos[i]-center[1],z_pos[i]-center[2]] for i in range(n)]

    dist = np.asarray(dist1)
    pos = np.asarray(pos1)


    r_vec = np.sqrt(np.sum(np.asarray(pos)**2,axis=1))
    r_unit_vec = np.sqrt(np.sum(np.asarray(pos)**2,axis=1))/dist

    pos_norm = np.asarray(pos)/np.asarray(dist)[:,None]


    Vol = sphere['dx'] * sphere['dy'] * sphere['dz'] 
    A = sphere['dx'] * sphere['dy'] #find better way to define area --------------------------
    R = (3 * Vol / (4 * np.pi))**(1/3) 

    angle = np.arctan2(R.v, r_vec)


    #NSIDE = 20
    NPIX = hp.nside2npix(nside)

    im = np.zeros(NPIX)

    for i in range(n):
        pixels = hp.query_disc(nside, pos_norm[i], angle[i], inclusive=True, )  
        for p in pixels:
            im[p] += quant[i]/len(pixels)


    

    if do_proj == True:
        im[pixels] = im.max()
        hp.mollview(im, title="Mollview image RING", return_projected_map=True)
        hp.graticule()
      
    #np.savetxt('file_radius_{}_center_{}.txt'.format(str(size_sphere), str(center)), im)
    
    #print(NPIX)    
    #print np.sum(im)
    
    return im

#get mass from this function -> sum over all pixels


def mass_sphere(path, size_sphere, c, dimensionless = True):
    '''
    Returns the total mass within a sphere and the volume of that sphere.
    
    Parameters
    -----------
    path          : the path to the simulation/snapshot file to be opened
    size_sphere   : the radius of the sphere within the domain to be analyzed
    c             : coordinate of the center of the sphere
    dimensionless : sets whether the output of the function is dimensionless or not (g/cm^3)
    
    
    Returns
    -----------
    The total mass within the sphere (in grams).
    The volume of the sphere (in cm^3).
    '''

    pc = 3.085e18

    ds = yt.load(path)

    center = c
    radius = (size_sphere, "pc")
    #sphere = ds.sphere(center, radius)

    sp = ds.sphere(c, radius)
    #V = (4*np.pi/3)*(size_sphere*pc*u.cm)**3
    V = (4*np.pi/3)*(size_sphere*pc)**3

    if dimensionless == True:
        return sp.quantities.total_quantity(["cell_mass"]).v, V
    else:
        V = (4*np.pi/3)*(size_sphere*pc*u.cm)**3
        return sp.quantities.total_quantity(["cell_mass"]).v*u.g, V 

    
    #def cover_dom_mp(path, rad, samples, nside):
def cover_dom_mp(path, rad, samples, nside, res, radius_in_pc=True):
    '''
    Returns the density and column density of a number of spheres within the domain sampled at random locations.
    
    Parameters
    -----------
    path         : the path to the simulation/snapshot file to be opened
    rad          : the radius of the sphere within the domain to be analyzed
    samples      : the number of samples within the domain
    nside        : parameter which determines the resolution of the map, usually a power of 2
    
    Returns
    -----------
    density      : average density within a sphere
    col_dens     : an array of column densities for each pixel within a sphere
    median_mp    : median of the column densities in a sphere
    avg_mp       : average of the column densities in a sphere
    centers      : coordinates of each sphere center in the domain
    
    
    
    
    
    '''
    
    f = path[-4:]

    ds = yt.load(path)
    
    prop = "cell_mass"
    pc = 3.085e18
    
    #creates coordinates for each axis, with one radius of "padding"
    ledge = ds.domain_left_edge
    redge = ds.domain_right_edge
    if radius_in_pc:
        radius = rad*pc
    else:
        radius = rad
    # do a sanity check here
    if radius > ds.domain_width[0].v or radius > ds.domain_width[1].v or radius > ds.domain_width[2].v:
        print("The radius you set is too large for the box")
    x = np.arange(ledge[0].v + radius, redge[0].v - radius, 2*radius)
    y = np.arange(ledge[1].v + radius, redge[1].v - radius, 2*radius)
    z = np.arange(ledge[2].v + radius, redge[2].v - radius, 2*radius)

    #creates "the cube"
    cube_simple = np.transpose(np.dstack((x,y,z)))

    pos = []
    
    #cartesian product of each point along each axis, creates every possible "center" and stores it in pos[0]
    pos.append(tuple(itertools.product(x,y,z)))

    #samples a pre-determined number of random centers within the domain (without replacement)
    #coords = random.sample(range(0, len(pos[0])), samples)
    
    #coords = random.choices(range(0, len(pos[0])), k=samples) #with replacement for Python 3.6 and newer
                            
    coords = [random.choice(range(0,len(pos[0]))) for _ in range(samples)]  #with replacement, for Python 3.5 and older
    
    
    
    
    #iterates through the previously created array of random coordinate centers
    centers = []
    for i in coords:
        centers.append(np.asarray(pos[0][i]))

    density = []
    col_dens = []

    k = 1
    for center in centers:
        total_mass, vol = mass_sphere(path, rad, center)
        density.append(total_mass/vol)
        mass = mollw_proj(path, rad, center, nside, prop)
        col_dens.append(mass)
        
        print("Sphere {} done".format(k))
        k+=1
    
    median_mp = []
    for i in range(len(col_dens)):
        median_mp.append(np.median(col_dens[i]))
    
    avg_mp = []
    for i in range(len(col_dens)):
        avg_mp.append(np.average(col_dens[i]))
        
    
        
    
    '''with open("data_{}_nside{}_4pc_R50_with_centers_true".format(str(samples), str(nside)), "wb") as data:
        pickle.dump({"density":np.asarray(density), "column_density": np.asarray(col_dens), 
                     "median": np.asarray(median_mp), "average": np.asarray(avg_mp), 
                     "centers": np.asarray(centers)}, data)'''
        
    
    #return density, col_dens, np.asarray(median_mp), np.asarray(avg_mp)
        
        
    with open("data_{}_nside{}_{}pc_R{}_{}".format(str(samples), str(nside), str(res), str(rad),f), "wb") as data:
        pickle.dump({"density":np.asarray(density), "column_density": np.asarray(col_dens),
                     "median": np.asarray(median_mp), "average": np.asarray(avg_mp),
                     "centers": np.asarray(centers)}, data)
        
def cover_dom_mpv(path, rad, samples, nside, res, radius_in_pc=True):
    '''
    Returns the density and column density of a number of spheres within the domain sampled at random locations.
    
    Parameters
    -----------
    path         : the path to the simulation/snapshot file to be opened
    rad          : the radius of the sphere within the domain to be analyzed
    samples      : the number of samples within the domain
    nside        : parameter which determines the resolution of the map, usually a power of 2
    
    Returns
    -----------
    density      : average density within a sphere
    col_dens     : an array of column densities for each pixel within a sphere
    median_mp    : median of the column densities in a sphere
    avg_mp       : average of the column densities in a sphere
    vel_disp     : velocity dispersion (units of km/s)
    centers      : coordinates of each sphere center in the domain
    
    
    
    
    
    '''
    
    f = path[-4:]
    
    prop = "cell_mass"
    pc = 3.085e18
    
    ds = yt.load(path)

    
    
    #creates coordinates for each axis, with one radius of "padding"
    ledge = ds.domain_left_edge
    redge = ds.domain_right_edge
    if radius_in_pc:
        radius = rad*pc
    else:
        radius = rad
    # do a sanity check here
    if radius > ds.domain_width[0].v or radius > ds.domain_width[1].v or radius > ds.domain_width[2].v:
        print("The radius you set is too large for the box")
    x = np.arange(ledge[0].v + radius, redge[0].v - radius, 2*radius)
    y = np.arange(ledge[1].v + radius, redge[1].v - radius, 2*radius)
    z = np.arange(ledge[2].v + radius, redge[2].v - radius, 2*radius)


    #creates "the cube"
    cube_simple = np.transpose(np.dstack((x,y,z)))

    pos = []
    
    #cartesian product of each point along each axis, creates every possible "center" and stores it in pos[0]
    pos.append(tuple(itertools.product(x,y,z)))

    #samples a pre-determined number of random centers within the domain (without replacement)
    #coords = random.sample(range(0, len(pos[0])), samples)
    
    #coords = random.choices(range(0, len(pos[0])), k=samples) #with replacement for Python 3.6 and newer
                            
    coords = [random.choice(range(0,len(pos[0]))) for _ in range(samples)]  #with replacement, for Python 3.5 and older
    
    
    
    
    #iterates through the previously created array of random coordinate centers
    centers = []
    for i in coords:
        centers.append(np.asarray(pos[0][i]))

    density = []
    col_dens = []
    vel_disp = []

    k = 1
    for center in centers:
        total_mass, vol = mass_sphere(path, rad, center)
        density.append(total_mass/vol)
        mass = mollw_proj(path, rad, center, nside, prop)
        col_dens.append(mass)
        
        sphere = ds.sphere(center, rad*pc)
        v_tot = np.sqrt(sphere['velx']*sphere['velx'] + sphere['vely']*sphere['vely'] + sphere['velz']*sphere['velz'])
        vel_disp.append(np.std(v_tot))
        
        print("Sphere {} done".format(k))
        k+=1
    
    median_mp = []
    for i in range(len(col_dens)):
        median_mp.append(np.median(col_dens[i]))
    
    avg_mp = []
    for i in range(len(col_dens)):
        avg_mp.append(np.average(col_dens[i]))
    
    
        
    
        
    
    '''with open("data_{}_nside{}_4pc_R50_with_centers_true".format(str(samples), str(nside)), "wb") as data:
        pickle.dump({"density":np.asarray(density), "column_density": np.asarray(col_dens), 
                     "median": np.asarray(median_mp), "average": np.asarray(avg_mp), 
                     "centers": np.asarray(centers)}, data)'''
        
    
    #return density, col_dens, np.asarray(median_mp), np.asarray(avg_mp)
        
        
    with open("datav_{}_nside{}_{}pc_R{}_{}".format(str(samples), str(nside), str(res), str(rad),f), "wb") as data:
        pickle.dump({"density":np.asarray(density), "column_density": np.asarray(col_dens),
                     "median": np.asarray(median_mp), "average": np.asarray(avg_mp), 
                     "vel_disp": np.asarray(vel_disp)/100000, "centers": np.asarray(centers)}, data)
        
        
def get_data(filename):
    '''
    Unpacks a dataset (pickle file) created with the function cover_dom_mp.
    
    
    Parameters
    ----------
    filename  : name of the pickle file containing the data
    
    
    Returns
    ----------
    den, col_den, median_mp, avg_mp, centers
    '''
    
    with open(filename, "rb") as all_data:
        data = pickle.load(all_data)
        
    den  = data['density'] 
    col_den = data['column_density']
    median_mp = data['median'] 
    avg_mp = data['average']
    centers = data['centers']
    
    
    return den, col_den, median_mp, avg_mp, centers


def create_plot(filename, p=False):
    '''
    Creates a plot of the log10 of the median column density within the sampled spheres as a 
    function of the log10 of density of those spheres.
    
    Parameters
    ----------
    filename : the name of the data file
    
    
    Returns
    ----------
    
    Plot of log10 of median column density within the sampled spheres as a 
    function of the log10 of density of those spheres.
    
    If p parameter set to True, also saves figure.
    
    '''
    
    pattern = r'\d*\.\d+|\d+'

    nums = re.findall(pattern, filename)
    
    with open(filename, "rb") as all_data:
        data = pickle.load(all_data)
        
    den  = data['density'] 
    col_den = data['column_density']
    median_mp = data['median'] 
    avg_mp = data['average']
    centers = data['centers']
    
    pc = 3.085e18
    rad = int(nums[3])
    nside = int(nums[1])
    A = 4 * np.pi * (rad*pc)**2

    #area of a pixel in cm^2
    A_cm = A/hp.nside2npix(nside)
    
    width = [] #difference between higher and lower percentile
    for i in range(len(col_den)):
        width.append(np.percentile(np.log10(col_den[i]/A_cm), 84) - np.percentile(np.log10(col_den[i]/A_cm), 16))
    

    
    plt.figure(figsize=(10,7.5))
    plt.scatter(np.log10(den), np.log10(median_mp/A_cm),c=width)
    plt.colorbar()
    plt.title('Time = {} Myrs'.format(int(nums[4])/100))
    plt.xlabel(r"$\rho_S$ [log10($g/cm^3$)]")
    plt.ylabel(r"$\Sigma_S$ log10($g/cm^2$)")
    plt.xlim(-28.5,-21)
    plt.ylim(-9,-3)
        
    
    if p == True:
        plt.savefig("nside{}_{}pc_R{}_{}.pdf".format(nums[1],nums[2], nums[3],nums[4]))
    plt.show()
    
def width_plot(filename, high = 84, low = 16, p = False):
    '''
    Creates a plot of the difference between selected percentiles (by default 84 and 16) 
    as a function of sphere density.
    
    Parameters
    ----------
    filename :
    high     :
    low      :
    
    Returns
    ----------
    Scatter plot
    
    
    
    '''
    
    plt.figure(figsize=(10,7.5))
    
    #pattern = r'[0-9]+'
    pattern = r'\d*\.\d+|\d+'

    nums = re.findall(pattern, filename)
    
    
    with open(filename, "rb") as all_data:
        data = pickle.load(all_data)
        
    den  = data['density'] 
    col_den = data['column_density']
    median_mp = data['median'] 
    avg_mp = data['average']
    centers = data['centers']

    width = [] #difference between higher and lower percentile
    for i in range(len(col_den)):
        width.append(np.percentile(np.log10(col_den[i]), high) - np.percentile(np.log10(col_den[i]), low))
        
    #fin = np.isfinite(np.asarray(width))
        
    plt.scatter(np.log10(np.asarray(den)), np.asarray(width))
    plt.ylabel(r'$\Sigma_{}^{}$ [log10($g/cm^2$)]'
               .format('{'+str(low)+'}','{'+str(high)+'}',high, low), wrap = True)
    plt.xlabel(r'$\rho_S$ [log10($g/cm^3$)]')
    plt.title('Time = {} Myrs'
              .format(int(nums[4])/100))
    plt.xlim(-28.5,-21)
    plt.ylim(top=4.8)
    
    
    if p == True:
        plt.savefig("width_nside{}_{}pc_R{}_{}.pdf".format(nums[1],nums[2], nums[3],nums[4]))

    plt.show()
    
def col_den_hist(filename, i, plot=False):
    '''
    
    Returns column density for a particular sphere in the domain.
    Optionally, creates histograms for that sphere.  
    
    '''
    
    with open(filename, "rb") as all_data:
        data = pickle.load(all_data)
        
    #den  = data['density'] 
    col_den = data['column_density']
    #median_mp = data['median'] 
    #avg_mp = data['average']
    
    if plot == True:
        plt.hist(np.log10(col_den[i]))
        plt.xlabel("Column density (log10)")

        
    return col_den[i]
    
def covering_fraction(filename, threshold):
    '''
    Returns covering fraction as a function of density in a sphere. Covering fraction is defined as
    how many lines of sight within a sphere have column density above a certain threshold divided by the 
    total lines of sight (i.e., pixels covering the sphere).
    
    Parameters:
    -----------
    filename :
    thershold:
    nside    :
    
    
    Returns:
    ----------
    den :
    cov :
    z   :

    '''

    #pattern = r'[0-9]+'
    pattern = r'\d*\.\d+|\d+'
    rad = int(re.findall(pattern, filename)[3])
    nside = int(re.findall(pattern, filename)[1])
    
    pc = 3.085e18
    A = 4 * np.pi * (rad*pc)**2

    #area of a pixel in cm^2
    A_cm = A/hp.nside2npix(nside) 

    n_H = 6e+23 #number of hydrogen atoms in one gram of gas


    with open(filename, "rb") as all_data:
        data = pickle.load(all_data)

    den  = data['density'] 
    col_den = data['column_density']
    median_mp = data['median'] 
    avg_mp = data['average']
    centers = data['centers']

    #particles per cm^2 
    b = col_den*n_H/A_cm
    '''cov = []
    for i in range(len(col_den)):
        above = np.where(b[i] > threshold)
        cov.append(len(b[above])/len(b))'''
    cov = []
    for i in range(len(col_den)):
        #above = np.where(b[i] > threshold)
        #cov.append(len(b[above])/len(b[0]))
        #cov.append(len(np.where(b[i] > threshold)[0])/len(b[0])) hp.nside2npix(nside)
        cov.append(len(np.where(b[i] > threshold)[0])/hp.nside2npix(nside))

    z = centers[:,2]
    
    return den, cov, z


def cov_plots(name):
    '''
    Covering fraction plots for various thresholds, starting from 1e+15 to 1e+20 particles cm^-2.
    
    
    '''
    with open(name, "rb") as all_data:
        data = pickle.load(all_data)

    den  = data['density'] 
    col_den = data['column_density']
    median_mp = data['median'] 
    avg_mp = data['average']
    centers = data['centers']
    
    den_part = den*6e+23
    
    pattern = r'\d*\.\d+|\d+'
    nside = int(re.findall(pattern, name)[1])

    
    th = 1e+17
    fig, ax =  plt.subplots(2,3, figsize=(20,15), sharex=True, sharey=True)
    t = 0
    for i in range(2):
        for j in range(3):
            den, cov, z = covering_fraction(name, th)
            ax[i][j].scatter(np.log10(den_part), cov)
            ax[i][j].set_title("{:.0e} particles per $cm^2$"
                               .format(th), wrap=True)
            ax[1][j].set_xlabel("Density [log10($particles/cm^3$)]")
            ax[i][0].set_ylabel("$\kappa$")

            th *= 10
            
def avg_width(filename, high = 75, low = 25):
    
    with open(filename, "rb") as all_data:
        data = pickle.load(all_data)
        
    den  = data['density'] 
    col_den = data['column_density']
    median_mp = data['median'] 
    avg_mp = data['average']
    centers = data['centers']
    

    width = [] #difference between higher and lower percentile
    for i in range(len(col_den)):
        width.append(np.percentile(np.log10(col_den[i]), high) - np.percentile(np.log10(col_den[i]), low))
        
    avgw = np.average(width)
    
    return avgw


def width_den(filename, high = 75, low = 25):
    
    with open(filename, "rb") as all_data:
        data = pickle.load(all_data)
        
    den  = data['density'] 
    col_den = data['column_density']
    median_mp = data['median'] 
    avg_mp = data['average']
    centers = data['centers']
    

 
    
    width = [] #difference between higher and lower percentile

    width.append(np.percentile(np.log10(den), high) - np.percentile(np.log10(den), low))

    return width


def sigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-a*(x-b)))


'''def avg_cd(filename):
    
    #Average column density in a snapshot. 
    
    
    
    with open(filename, "rb") as all_data:
        data = pickle.load(all_data)
        
    den  = data['density'] 
    col_den = data['column_density']
    median_mp = data['median'] 
    avg_mp = data['average']
    centers = data['centers']
    

    avg_cd = [] 
    for i in range(len(col_den)):
        avg_cd.append(np.average(np.log10(col_den[i])))
        
    avg_cd = np.average(avg_cd)
    
    return avg_cd'''

def avg_cd(filename):
    '''
    Average column density in a snapshot. 
    
    '''
    pattern = r'\d*\.\d+|\d+'

    nums = re.findall(pattern, filename)
    
    pc = 3.085e18
    rad = int(nums[3])
    nside = int(nums[1])
    A = 4 * np.pi * (rad*pc)**2

    #area of a pixel in cm^2
    A_cm = A/hp.nside2npix(nside)
    
    with open(filename, "rb") as all_data:
        data = pickle.load(all_data)
        
    den  = data['density'] 
    col_den = data['column_density']
    median_mp = data['median'] 
    avg_mp = data['average']
    centers = data['centers']
    

    avg_cd = [] 
    for i in range(len(col_den)):
        #cd = col_den[i][col_den[i] != 0.0]
        cd = np.where(col_den[i] == 0,np.average(col_den[i]), col_den[i]) #excludes columns with 0 column density and
        avg_cd.append(np.log10(cd/A_cm))                                       #replaces them with the average value
        
    avg = np.average(avg_cd)
    
    return avg


def avg_den(filename):
    '''
    Average density in a snapshot.
    '''
    
    with open(filename, "rb") as all_data:
        data = pickle.load(all_data)
        
    den  = data['density'] 
    col_den = data['column_density']
    median_mp = data['median'] 
    avg_mp = data['average']
    centers = data['centers']
    

 
    
    avg_den = [] #difference between higher and lower percentile
    for i in range(len(den)):
        avg_den.append(np.log10(den))
    
    avg_den = np.mean(avg_den)

    return avg_den



def cov_plot_fit(name, th=1e18):
    '''
    Creates a plot of covering fraction with sigmoid function fit.
    
    '''


    with open(name, "rb") as all_data:
            data = pickle.load(all_data)

    den  = data['density'] 
    col_den = data['column_density']
    median_mp = data['median'] 
    avg_mp = data['average']
    centers = data['centers']

    den_part = den*6e+23

    pattern = r'\d*\.\d+|\d+'
    nside = int(re.findall(pattern, name)[1])
    res = float(re.findall(pattern, name)[2])
    snap = int(re.findall(pattern, name)[-1])

    #th = 1e18    
    plt.figure(figsize=(10,7.5))

    den, cov, z = covering_fraction(name, th)
    plt.scatter(np.log10(den_part), cov, alpha=0.2, label="Data")
    plt.title("[Snapshot {}, resolution {} pc] Covering fraction as a function of density \n with threshold {:.0e} particles per cm^2"
                       .format(snap, res, th), wrap=True)
    #ax[1][1].set_xlabel("Density (log10 $particles/cm^3$)")
    #ax[0][0].set_ylabel("Covering fraction")
    
    
    
    bin_means, bin_edges, binnumber = stats.binned_statistic(np.log10(den_part),
                    cov, statistic=np.mean, bins=10)

    bin_std, bin_edges1, binnumber1 = stats.binned_statistic(np.log10(den_part),
                    cov, statistic=np.std, bins=10)
    
    bin_mid = []
    for j in range(len(bin_edges)-1):
        bin_mid.append((bin_edges[j]+bin_edges[j+1])/2)

        
    #turn scatter plot into line plot (when data is not in any particular order)
    #see #https://stackoverflow.com/questions/37414916/pythons-matplotlib-plotting-in-wrong-order

    
    plt.scatter(bin_mid, bin_means, s=50, label="Grouped points")
    popt, pcov1 = curve_fit(sigmoid, np.log10(den_part), cov)

    x = np.log10(den_part)
    y = sigmoid(np.log10(den_part), *popt)

    lists = sorted(zip(*[x, y]))
    new_x, new_y = list(zip(*lists))


    plt.errorbar(bin_mid, bin_means, yerr=bin_std, c='orange', fmt=" ")
    #plt.scatter(np.log10(den_part), sigmoid(np.log10(den_part), *popt), label="Fit to data")

    plt.plot(new_x,new_y,c='green',lw=4,label="Fit to data")
    


    print(popt)
    #plt.scatter(np.log10(den_part), sigmoid(np.log10(den_part), *popt1))
    #plt.scatter(np.log10(den_part), t(np.log10(den_part), *popth))
    #plt.scatter(np.log10(den_part), sigmoid(np.log10(den_part), *popt))

    plt.xlabel("Density (log10 $particles/cm^3$)")
    plt.ylabel("Covering fraction")
    plt.legend()
    
    return popt, pcov1
    
    
    
def avg_den_plots(*args, shapes=False):
    '''
    Creates a plot of average density in a snapshot as a function of the snapshot name. Takes as argument as many as
    seven file names and outputs a single plot color coded for a specific sphere radius and nside parameter.
    '''

    #for label/legend issue look here 
    #https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib/40870637

    age = [0,10,20,30,40,50,60] #the time for different snaps in the sim (different for 1pc and 2pc)
    
    l = len(args[0])      #a clunky way to ensure the plots show the right sim time (see scatter plot)
    a = 7-len(args[0])    #this is to distinguish between 1pc and 2pc simulations, which have different # of snaps

    plt.figure(figsize=(10,7.5))

    plt.figure(figsize=(10,7.5))

    colors = ['r','b','k','g','c','m',['orange']] #colors written with by multiple letters don't work in for loop
                                                  #must be put in separate list
    i = 0 
    j = 0
    
    sym = 'x'

    for names in args:
        #for name,c in zip(names,len(names)*colors[i]):
        for name in names:
            pattern = r'\d*\.\d+|\d+'
            spheres = int(re.findall(pattern, name)[0])
            nside = int(re.findall(pattern, name)[1])
            resolution = float(re.findall(pattern, name)[2])
            radius = int(re.findall(pattern, name)[3])
            snap = int(re.findall(pattern, name)[4])
            
            if radius == 10:
                c = 'r'
            elif radius == 25:
                c = 'b'
            elif radius == 50:
                c = 'k'

            if shapes == False:
                sym = "o"
                
            elif shapes == True:
                if nside == 4:
                    sym = 's'
                elif nside == 8:
                    sym = 'o'
                elif nside == 16:
                    sym = 'D'

            
                
            
            plt.scatter(age[a+j%l], avg_den(name),color=c, marker = sym, label="R={}pc".format(radius) if name == names[0] else "")
            plt.xlabel("Time (Myrs)")
            plt.ylabel(r"$\rho$ [log10($g/cm^3$)]")
            plt.legend()
            j+=1
        i+=1
         

def avg_cd_plots(*args,shapes=False):
    '''
    Creates a plot of average column density in a snapshot as a function of the snapshot name. 
    Also returns the average column density.
    '''
    
    #for label/legend issue look here 
    #https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib/40870637

    
    age = [0,10,20,30,40,50,60] #the time for different snaps in the sim (different for 1pc and 2pc)
    
    l = len(args[0])      #a clunky way to ensure the plots show the right sim time (see scatter plot)
    a = 7-len(args[0])    #this is to distinguish between 1pc and 2pc simulations, which have different # of snaps

    plt.figure(figsize=(10,7.5))

    colors = ['r','b','k','g','c','m',['orange']] #colors written with by multiple letters don't work in for loop
                                                  #must be put in separate list
    i = 0                                         
    j = 0
    avgcd = []
    
    sym = 'x'
    
    
    for names in args:
        for name,c in zip(names,len(names)*colors[i]):
            pattern = r'\d*\.\d+|\d+'
            spheres = int(re.findall(pattern, name)[0])
            nside = int(re.findall(pattern, name)[1])
            resolution = float(re.findall(pattern, name)[2])
            radius = int(re.findall(pattern, name)[3])
            snap = int(re.findall(pattern, name)[4])
            
            if shapes == False:
                sym = "o"
                
            elif shapes == True:
                if nside == 4:
                    sym = 's'
                elif nside == 8:
                    sym = 'o'
                elif nside == 16:
                    sym = 'D'
            

            plt.scatter(age[a+j%l], avg_cd(name),color=c, marker=sym, label="R={}pc".format(radius,nside) if name == names[0] else "")
            plt.xlabel("Time (Myrs)")
            plt.ylabel(r"Average column density [log10($g/cm^2$)]")
            plt.legend()
            print("R={}pc nside= {} has avg cd {}".format(radius,nside, avg_cd(name)))
            avgcd.append((radius,nside,avg_cd(name)))
            j+=1
        i+=1
        
    return avgcd
        
def cov_plot_fit1(name, th=1e18):
    '''
    Creates a plot of covering fraction with sigmoid function fit.
    Returns popt and perr.
    
    '''


    with open(name, "rb") as all_data:
            data = pickle.load(all_data)

    den  = data['density'] 
    col_den = data['column_density']
    median_mp = data['median'] 
    avg_mp = data['average']
    centers = data['centers']

    den_part = den*6e+23

    pattern = r'\d*\.\d+|\d+'
    nside = int(re.findall(pattern, name)[1])
    res = int(re.findall(pattern, name)[2])
    snap = int(re.findall(pattern, name)[-1])

    #th = 1e18    
    plt.figure(figsize=(10,7.5))

    den, cov, z = covering_fraction(name, th)
    plt.scatter(np.log10(den_part), cov, alpha=0.2, label="Data")
    plt.title("[Snapshot {}, resolution {} pc] Covering fraction as a function of density \n with threshold {:.0e} particles per cm^2"
                       .format(snap, res, th), wrap=True)
    #ax[1][1].set_xlabel("Density (log10 $particles/cm^3$)")
    #ax[0][0].set_ylabel("Covering fraction")
    
    
    
    bin_means, bin_edges, binnumber = stats.binned_statistic(np.log10(den_part),
                    cov, statistic=np.mean, bins=10)

    bin_std, bin_edges1, binnumber1 = stats.binned_statistic(np.log10(den_part),
                    cov, statistic=np.std, bins=10)
    
    bin_mid = []
    for j in range(len(bin_edges)-1):
        bin_mid.append((bin_edges[j]+bin_edges[j+1])/2)

        
    #turn scatter plot into line plot (when data is not in any particular order)
    #see #https://stackoverflow.com/questions/37414916/pythons-matplotlib-plotting-in-wrong-order

    
    plt.scatter(bin_mid, bin_means, s=50, label="Grouped points")
    popt, pcov1 = curve_fit(sigmoid, np.log10(den_part), cov)

    x = np.log10(den_part)
    y = sigmoid(np.log10(den_part), *popt)

    lists = sorted(zip(*[x, y]))
    new_x, new_y = list(zip(*lists))


    plt.errorbar(bin_mid, bin_means, yerr=bin_std, c='orange', fmt=" ")
    #plt.scatter(np.log10(den_part), sigmoid(np.log10(den_part), *popt), label="Fit to data")

    plt.plot(new_x,new_y,c='green',lw=4,label="Fit to data")
    

    perr = np.sqrt(np.diag(pcov1))
    
    #print(popt)
    #print(perr)
    #plt.scatter(np.log10(den_part), sigmoid(np.log10(den_part), *popt1))
    #plt.scatter(np.log10(den_part), t(np.log10(den_part), *popth))
    #plt.scatter(np.log10(den_part), sigmoid(np.log10(den_part), *popt))

    plt.xlabel("Density (log10 $particles/cm^3$)")
    plt.ylabel("Covering fraction")
    plt.legend()
    
    return popt, perr


def diff_res(*args):
    '''
    Returns difference
    
    '''
    n = len(args) #number of arguments
    avgcd = avg_cd_plots(*args)

    avgcd = np.asarray(avgcd)

    #col_dens = avgcd[:,2].reshape(3,int(len(avgcd[:,2])/n))
    #col_dens = avgcd[:,2].reshape(int(len(avgcd[:,2])/n),3)
    
    col_dens = avgcd[:,2].reshape(n,int(len(avgcd)/n))


    a = []
    for i in range(len(col_dens[0])):
        print(10**(col_dens[0][i]-col_dens[n-1][i]))
        a.append(10**(col_dens[0][i]-col_dens[n-1][i]))

    avg = sum(a)/len(a)
    
    return avg
