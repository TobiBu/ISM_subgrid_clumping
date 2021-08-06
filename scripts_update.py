

import yt
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import itertools
import random
from yt.utilities.math_utils import get_cyl_theta, get_cyl_theta_component, euclidean_dist
import pickle


def get_age(path):

	ds = yt.load(path)
	print("Time of sim is: {} Myrs ".format(ds.current_time.in_units("Myr")))


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
    
    return im,sphere["cell_mass"].v,sphere["cell_volume"].v





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



def cover_dom(path, rad, samples):
    '''
    Returns a list of average densities in spheres of a certain radius within the domain. 
    Creates a list of random positions within the domain and calculates the density of spheres centered at
    those positions using the mass_sphere function defined previously. 

    Parameters:
    -----------
    path         : the path to the simulation/snapshot file to be opened
    rad          : the radius of the sphere within the domain to be analyzed
    samples      : the number of samples within the domain

    Returns:
    -----------
    density_x, ensity_y, density_z: lists containing the densities of spheres along the edges of the domain.

    '''
    
    pc = 3.085e18
    
    #creates coordinates for each axis
    x = np.arange(-7.715e+20 + rad*pc, 7.715e+20 - rad*pc, 2*rad*pc)
    y = np.arange(-7.715e+20 + rad*pc, 7.715e+20 - rad*pc, 2*rad*pc)
    z = np.arange(-7.715e+20 + rad*pc, 7.715e+20 - rad*pc, 2*rad*pc)

    #creates "the cube"
    cube_simple = np.transpose(np.dstack((x,y,z)))

    pos = []
    
    #cartesian product of each point along each axis, creates every possible "center" 
    pos.append(tuple(itertools.product(x,y,z)))

    #samples a pre-determined number of random centers within the domain
    coords = random.sample(range(0, len(pos[0])), samples) #without replacement
    #coords = random.choices(range(0, len(pos[0])), k=samples) #with replacement, only works for Python 3.6 and newer
    #coords = [random.choice(range(0,len(pos[0]))) for _ in range(samples)] #with replacement, works for Python 3.5 and older

    centers = []
    for i in coords:
        centers.append(np.asarray(pos[0][i]))

    density = []

    for center in centers:
        total_mass, vol = mass_sphere(p, rad, center)
        density.append(total_mass/vol)
    
    return density




        
        
def cover_dom_mp(path, rad, samples, nside, res):
    '''
    Returns the density and column density of a number of spheres within the domain sampled at random locations.
    
    Parameters
    -----------
    path         : the path to the simulation/snapshot file to be opened
    rad          : the radius of the sphere within the domain to be analyzed
    samples      : the number of samples within the domain
    nside        : parameter which determines the resolution of the map, usually a power of 2
    res          : resolution of the simulation (for file naming purpose only) 
    Returns
    -----------
    density      : average density within a sphere
    col_dens     : an array of column densities for each pixel within a sphere
    median_mp    : median of the column densities in a sphere
    avg_mp       : average of the column densities in a sphere
    centers      : coordinates of each sphere center in the domain
    
    
    
    
    
    '''
    
    f = path[-4:]
    
    prop = "cell_mass"
    pc = 3.085e18
    
    
    #creates coordinates for each axis
    x = np.arange(-7.715e+20 + rad*pc, 7.715e+20 - rad*pc, 2*rad*pc)
    y = np.arange(-7.715e+20 + rad*pc, 7.715e+20 - rad*pc, 2*rad*pc)
    z = np.arange(-7.715e+20 + rad*pc, 7.715e+20 - rad*pc, 2*rad*pc)

    #creates "the cube"
    cube_simple = np.transpose(np.dstack((x,y,z)))

    pos = []
    
    #cartesian product of each point along each axis, creates every possible "center" 
    pos.append(tuple(itertools.product(x,y,z)))

    #samples a pre-determined number of random centers within the domain
    coords = random.sample(range(0, len(pos[0])), samples) #without replacement
    #coords = random.choices(range(0, len(pos[0])), k=samples)  #with replacement
    #coords = [random.choice(range(0,len(pos[0]))) for _ in range(samples)] #with replacement, works for Python 3.5 and older


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
        
    
    with open("data_{}_nside{}_{}pc_R{}_{}".format(str(samples), str(nside), str(res), str(rad),f), "wb") as data:
        pickle.dump({"density":np.asarray(density), "column_density": np.asarray(col_dens), 
                     "median": np.asarray(median_mp), "average": np.asarray(avg_mp), 
                     "centers": np.asarray(centers)}, data)    


def cover_dom_mpv(path, rad, samples, nside, res):
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
    density      : average density within a sphere (g/cm^3)
    col_dens     : an array of column densities for each pixel within a sphere (g/pixel)
    median_mp    : median of the column densities in a sphere    (g/pixel)
    avg_mp       : average of the column densities in a sphere   (g/pixel)
    vel_disp     : velocity dispersion (units of km/s)
    centers      : coordinates of each sphere center in the domain
    
    
    
    
    
    '''
    
    f = path[-4:]
    
    prop = "cell_mass"
    pc = 3.085e18

    ds = yt.load(path)    
    
    #creates coordinates for each axis, with one radius of "padding"
    x = np.arange(-7.715e+20 + rad*pc, 7.715e+20 - rad*pc, 2*rad*pc)
    y = np.arange(-7.715e+20 + rad*pc, 7.715e+20 - rad*pc, 2*rad*pc)
    z = np.arange(-7.715e+20 + rad*pc, 7.715e+20 - rad*pc, 2*rad*pc)

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



def cover_dom_mpv_overlap(path, rad, samples, nside, res):
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
    cell_masses  : the masses of individual cells in each sphere (grams)
    cell_volumes : the volumes of individual cells in each sphere (cm^3)
    centers      : coordinates of each sphere center in the domain
    
    
    
    
    
    '''
    
    f = path[-4:]
    
    prop = "cell_mass"
    pc = 3.085e18
    
    ds = yt.load(path)

    

    #samples a pre-determined number of random centers within the domain (without replacement)
    #coords = random.sample(range(0, len(pos[0])), samples)
    
    #coords = random.choices(range(0, len(pos[0])), k=samples) #with replacement for Python 3.6 and newer
                            
    #creates coordinates for each axis, with one radius of "padding"
    x = np.arange(-7.715e+20 + rad*pc, 7.715e+20 - rad*pc, 2*pc)
    y = np.arange(-7.715e+20 + rad*pc, 7.715e+20 - rad*pc, 2*pc)
    z = np.arange(-7.715e+20 + rad*pc, 7.715e+20 - rad*pc, 2*pc)

    cube_simple = np.transpose(np.dstack((x,y,z)))

    pos = []

    #cartesian product of each point along each axis, creates every possible "center" and stores it in pos[0]
    pos.append(tuple(itertools.product(x,y,z)))

    coords = [random.choice(range(0,len(pos[0]))) for _ in range(samples)]  #with replacement, for Python 3.5 and older
    

    
    
    #iterates through the previously created array of random coordinate centers
    centers = []
    for i in coords:
        centers.append(np.asarray(pos[0][i]))

    density = []
    col_dens = []
    vel_disp = []

    cell_masses = []
    cell_volumes = []

    k = 1
    for center in centers:
        total_mass, vol = mass_sphere(path, rad, center)
        density.append(total_mass/vol)
        mass,masses,volumes = mollw_proj(path, rad, center, nside, prop)
        col_dens.append(mass)
        
        cell_masses.append(masses)
        cell_volumes.append(volumes)
        
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
    
    
        
    
        
    

        
    
    #return density, col_dens, np.asarray(median_mp), np.asarray(avg_mp)
        
        
    with open("data_cellprop_{}_nside{}_{}pc_R{}_{}".format(str(samples), str(nside), str(res), str(rad),f), "wb") as data:
        pickle.dump({"density":np.asarray(density), "column_density": np.asarray(col_dens),
                     "median": np.asarray(median_mp), "average": np.asarray(avg_mp), 
                     "vel_disp": np.asarray(vel_disp)/100000, "cell_masses": np.asarray(cell_masses),"cell_volumes": np.asarray(cell_volumes), 
                      "centers": np.asarray(centers)}, data)


def get_data(filename):
    '''
    Unpacks a dataset (pickle file) created with the function cover_dom_mp.
    
    
    Parameters
    ----------
    filename  : name of the pickle file containing the data
    
    
    Returns
    ----------
    den, col_den, median_mp, avg_mp, cell_masses, cell_volumes, centers
    '''
    
    with open(filename, "rb") as all_data:
        data = pickle.load(all_data)
        
    den          = data['density'] 
    col_den      = data['column_density']
    median_mp    = data['median'] 
    avg_mp       = data['average']
    cell_masses  = data['cell_masses']
    cell_volumes = data['cell_volumes']
    centers      = data['centers']
    
    
    return den, col_den, median_mp, avg_mp, cell_masses, cell_volumes, centers




def create_plot(filename):
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
    
    '''
    
    with open(filename, "rb") as all_data:
        data = pickle.load(all_data)
        
    den  = data['density'] 
    col_den = data['column_density']
    median_mp = data['median'] 
    avg_mp = data['average']
    centers = data['centers']
    
    plt.scatter(np.log10(den), np.log10(median_mp))
    plt.xlabel("Density")
    plt.ylabel("Column density")
    plt.show()




def width_plot(filename, high = 75, low = 25):
    '''
    Creates a plot of the difference between selected percentiles (by default 75 and 25) 
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
        
    fin = np.isfinite(np.asarray(width))
        
    plt.scatter(np.log10(np.asarray(den)[fin]), np.asarray(width)[fin])
    plt.ylabel("Difference between {} percentile and {} percentile column within a sphere".format(high, low))
    plt.xlabel("Average density within the shere")
    plt.show()





def col_den_hist_all(filename):
    '''
    Creates histograms for all the spheres in a dataset. Returns an array containing all the column densities of 
    the sphere within the domain.
    '''
    
    with open(filename, "rb") as all_data:
        data = pickle.load(all_data)
        
    den  = data['density'] 
    col_den = data['column_density']
    median_mp = data['median'] 
    avg_mp = data['average']
    centers = data['centers']
    
    for i in range(0,len(den),1):
        try:
            plt.hist(np.log10(col_den[i]), label="Column density")
            plt.savefig("histograms_automatic1/Histogram_{}.png".format(str(i)))
            plt.clf()
        except:
            print("Histogram_{} could not be printed".format(str(i)))
    
    return col_den





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
        try:
            plt.hist(np.log10(col_den[i]))
            plt.xlabel("Column density")
        except:
            print("Histogram could not be printed.")
        
    return col_den
    


def w_den_plot(path,res):    
    ds = yt.load(path)

    f = path[-4:]

    dd = ds.all_data()

    rho = dd["density"].v

    v_ = dd["cell_volume"]

    w_rho = np.sum(np.log10(rho)*v_)/np.sum(v_)

    plt.figure(figsize=(10,7.5))


    plt.hist(np.log10(rho), weights=v_, histtype='step', lw=5, color='b',density=True, label="Volume weighted density")
    

    plt.hist(np.log10(rho), histtype='step', color='orange',lw=5,density=True,label="Density")
    plt.axvline(np.mean(np.log10(rho)), color='orange',label="Mean density")
    plt.xlabel("Density log10(g/cm^3)")
    plt.axvline(w_rho,label="Volume weighted mean density")
    plt.legend()
    plt.savefig("Plot_{}pc_{}.pdf".format(res,f))
