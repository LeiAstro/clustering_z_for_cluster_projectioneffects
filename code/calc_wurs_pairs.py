#!/usr/bin/env python3

import gc
import logging
import timeit
from multiprocessing import cpu_count

import h5py  
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy import units as u
from joblib import Parallel, delayed
# from tqdm import tqdm
import treecorr

import yaml

import os

# Set NumExpr to use one thread per process to prevent oversubscription
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['NUMEXPR_MAX_THREADS'] = '4'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

yml_fname = "calc_wur_inputs.yml"
with open(yml_fname, 'r') as stream:
    try:
        para = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

args_npatches = para['npatches']
args_h0       = para['H0']
args_Om0      = para['Om0']

n_CPUs        = para['N_CPUs']

##########################
### For unknown sample ###
##########################

unk_path  = para['cluster_mem_path']
unk_fname = para['cluster_mem_fname']

richness_min = para['richness_min']
richness_max = para['richness_max']

############################
### For reference sample ###
############################

ref_path       = para['reference_path']
ref_fname_gax  = para['reference_gname']
ref_fname_rand = para['reference_rname']

##########################
### Set wide tomo bins ###
##########################

zwide_tomo_start = para['zwide_start']
zwide_tomo_end   = para['zwide_end']
dz_zwide         = para['wide_dz']
num_zwide_bins   = para['num_zwide_bins']

zwide_bins = np.linspace(zwide_tomo_start, 
                         zwide_tomo_end,
                         num_zwide_bins+1)

zmin_i_arr = zwide_bins[:-1] #[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
zmax_i_arr = zwide_bins[1:] #[0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

###########################
### Set narrow tomo bin ###
###########################

dz_i = para['narrow_tomo_dz']
logging.info(f'There are {float(dz_zwide/dz_i):.0f} narrow tomo bins in a wide tomo bin.')

print('The wide tomo bin:', zwide_bins)
print('The wide tomo bin size:', dz_zwide)
print('The narrow tomo bin size:', dz_i)
print('zmin_i_arr:', zmin_i_arr)
print('zmax_i_arr:', zmax_i_arr)

#################################
### Set reference "zspec" bin ###
#################################

dz_j   = para['spec_dz']
zmin_j = para['zspec_min']
zmax_j = para['zspec_max']

print('The spectroscopic redshift (reference) bin size:', dz_j)

#################################
### Set angular diameter bins ###
#################################

# Define aperture bins in physical Mpc distance
rmin = para['rmin']  #0.1  # Mpc
rmax = para['rmax']  #10.0  # Mpc
num_rbins = para['num_rbins']  #9
rbin_spacing = para['spacing']


cosmo = FlatLambdaCDM(
    H0 = args_h0  * u.km / u.s / u.Mpc,
    Om0= args_Om0,
    Tcmb0 = 2.725 * u.K
)

def micro_bins(z_min, z_max, dz, cosmo):
    """
    Build reference‑sample bins and their comoving distances.

    Parameters
    ----------
    cosmo : astropy.cosmology.FlatLambdaCDM
        Pre‑built cosmology object (re‑used everywhere).
    """
    jbins      = np.arange(z_min, z_max + dz, dz)
    zcen       = 0.5 * (jbins[:-1] + jbins[1:])
    dcen_mpc   = cosmo.comoving_distance(zcen).value  # ← single vectorised call
    return jbins, zcen, dcen_mpc


def rbins_phydis(rmin, rmax, nbins, spacing="linear"):
    """
    Generate physical‑scale radial bins in [rmin, rmax] [Mpc].

    Parameters
    ----------
    rmin, rmax : float
        Minimum and maximum radius (proper Mpc).
    nbins : int
        Number of bins.
    spacing : {'linear', 'log'}, optional
        'linear' (default) → equal‑width bins in r;
        'log'           → equal‑width bins in log₁₀(r).

    Returns
    -------
    rbins : ndarray, shape (nbins+1,)
        Bin edges.
    dr_bins : ndarray, shape (nbins,)
        Bin widths  (rbins[1:] − rbins[:-1]).
        Always returned as an array so it works for either spacing.
    rbins_cen : ndarray, shape (nbins,)
        Bin centres 0.5 × (edge_i + edge_{i+1}).
    """
    if spacing == "linear":
        rbins = np.linspace(rmin, rmax, nbins + 1)
    elif spacing == "log":
        if rmin <= 0:
            raise ValueError("rmin must be > 0 for logarithmic bins.")
        rbins = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
    else:
        raise ValueError("spacing must be 'linear' or 'log'")

    dr_bins    = np.diff(rbins)                     # width per bin
    rbins_cen  = 0.5 * (rbins[:-1] + rbins[1:])    # centre per bin
    return rbins, dr_bins, rbins_cen


def load_data(richness_min=None, richness_max=None):
    """
    Load the required datasets efficiently, accommodating only a cluster member catalog.

    Returns:
    - data: dictionary containing loaded data arrays
    """
    data = {}
    try:
        # Load cluster members
        # pathz = '/lustre/work/client/users/lyang4/borah_scratch/data/redmapper_run_chisq_max_8/'
        # member_filename = 'newmatched_zspec_median_clusters_chisq_max_8_Cardinal-3Y6a_run_redmapper_v0.8.5_lgt20_vl02_catalog_members_p078.fit'
        fname_unk = unk_path + unk_fname

        with fits.open(fname_unk, memmap=True) as hdul:
            hdul.verify('fix')
            data_members = hdul[1].data
        if richness_min is not None and richness_max is not None:
            selz_tag = (data_members['lambda']>=richness_min)&(data_members['lambda']<richness_max)
            # Extract data from members
            member_cluster_id = data_members['mem_match_id'][selz_tag]  # cluster_id
            member_ra = data_members['ra'][selz_tag].astype(np.float64)
            member_dec = data_members['dec'][selz_tag].astype(np.float64)
            member_z_clus = data_members['zspec_cen_mem'][selz_tag].astype(np.float64)
        else:
            member_cluster_id = data_members['mem_match_id']  # cluster_id
            member_ra = data_members['ra'].astype(np.float64)
            member_dec = data_members['dec'].astype(np.float64)
            member_z_clus = data_members['zspec_cen_mem'].astype(np.float64)
            
        logging.info('Read in cluster members.')

        # Get unique cluster_ids
        cluster_ids = np.unique(member_cluster_id)

        # For each cluster_id, get z_lambda (should be the same for all members)
        cluster_z_center = []
        for cid in cluster_ids:
            idx = member_cluster_id == cid
            z_lambda_cid = np.unique(member_z_clus[idx])
            if len(z_lambda_cid) != 1:
                logging.warning(f"Multiple z_lambda for cluster_id {cid}")
            cluster_z_center.append(z_lambda_cid[0])

        data['member_cluster_id'] = member_cluster_id
        data['member_ra'] = member_ra
        data['member_dec'] = member_dec
        data['cluster_ids'] = cluster_ids
        data['cluster_z_center'] = np.array(cluster_z_center).astype(np.float64)

        # Clean up
        del data_members
        gc.collect()

        # Load Reference (galaxies)
        # refpath = #'/lustre/work/client/users/lyang4/borah_scratch/data/Cardinal/'
        filename_ref = ref_path + ref_fname_gax #'selELGs_cardinal_Alam2020.fits'
        with fits.open(filename_ref, memmap=True) as hdul:
            hdul.verify('fix')
            data_ref = hdul[1].data
        sel_zgal = data_ref['Z'] > 0
        data['ra_ref'] = data_ref['RA'][sel_zgal].astype(np.float64)
        data['dec_ref'] = data_ref['DEC'][sel_zgal].astype(np.float64)
        data['zspec_ref'] = data_ref['Z'][sel_zgal].astype(np.float64)
        logging.info('Read in reference galaxies.')

        del data_ref
        gc.collect()

        # Load reference randoms
        fname_ref_rand = ref_path + ref_fname_rand 
        with fits.open(fname_ref_rand, memmap=True) as hdul:
            hdul.verify('fix')
            rdata_ref = hdul[1].data
        sel_zrand = rdata_ref['Z'] > 0
        data['ra_ref_rand'] = rdata_ref['RA'][sel_zrand].astype(np.float64)
        data['dec_ref_rand'] = rdata_ref['DEC'][sel_zrand].astype(np.float64)
        data['zspec_ref_rand'] = rdata_ref['Z'][sel_zrand].astype(np.float64)
        logging.info('Read in reference randoms.')

        del rdata_ref
        gc.collect()

        # Data verification for cluster member arrays
        member_lengths = [
            len(data['member_cluster_id']),
            len(data['member_ra']),
            len(data['member_dec'])
        ]
        if not all(length == member_lengths[0] for length in member_lengths):
            logging.error("Mismatch in lengths of cluster member arrays.")
            return None

        # Data verification for cluster arrays
        cluster_lengths = [
            len(data['cluster_ids']),
            len(data['cluster_z_center'])
        ]
        if not all(length == cluster_lengths[0] for length in cluster_lengths):
            logging.error("Mismatch in lengths of cluster arrays.")
            return None

        return data

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return None
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None


def assign_jk_labels(ra_list, dec_list, npatches):
    """
    Generate spatial Jackknife labels using treecorr KMeans clustering.

    Parameters:
    - ra_list: list of RA arrays
    - dec_list: list of Dec arrays
    - npatches: number of Jackknife patches

    Returns:
    - labels_list: list of label arrays corresponding to input RA/Dec arrays
    """
    # Verify that ra_list and dec_list have matching lengths
    if len(ra_list) != len(dec_list):
        logging.error("ra_list and dec_list must have the same length.")
        return None

    all_ra = np.concatenate(ra_list)
    all_dec = np.concatenate(dec_list)

    # Check for matching lengths of concatenated arrays
    if len(all_ra) != len(all_dec):
        logging.error("Concatenated RA and Dec arrays must have the same length.")
        return None

    cat = treecorr.Catalog(ra=all_ra, dec=all_dec, ra_units='deg', dec_units='deg')
    field = cat.getNField()
    logging.info("Running KMeans clustering for Jackknife labels...")
    all_labels = field.run_kmeans(npatches)[0]

    labels_list = []
    start = 0
    for ra in ra_list:
        end = start + len(ra)
        labels = all_labels[start:end]
        labels_list.append(labels)
        start = end

    return labels_list


def compute_wDuDr_wDuRr(cluster_members, galaxies, randoms, rbins, dr_bins, rbins_cen, ang_diam_dist):
    """
    Compute wDuDr and wDuRr between cluster members and galaxies/randoms.

    Parameters:
    - cluster_members: dict with 'ra' and 'dec' of cluster members
    - galaxies: dict with 'ra' and 'dec' of galaxies
    - randoms: dict with 'ra' and 'dec' of random points
    - rbins: array of radial bin edges
    - dr_bins: bin width
    - rbins_cen: array of radial bin centers
    - ang_diam_dist: angular diameter distance at z_j

    Returns:
    - wDuDr: weighted pair counts for data-data
    - wDuRr: weighted pair counts for data-random
    - Nr: number of galaxies
    - Rr: number of randoms
    """
    from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks

    if len(galaxies['ra']) > 10 and len(cluster_members['ra']) > 0:
        # Convert physical rbins to angular scales using comoving distance
        theta_bins = np.degrees(rbins / ang_diam_dist).astype(np.float64)  # in degrees

        # Prepare data for pair counts
        ra_galaxies = galaxies['ra'].astype(np.float64)
        dec_galaxies = galaxies['dec'].astype(np.float64)

        ra_randoms = randoms['ra'].astype(np.float64)
        dec_randoms = randoms['dec'].astype(np.float64)

        ra_cluster_members = cluster_members['ra'].astype(np.float64)
        dec_cluster_members = cluster_members['dec'].astype(np.float64)

        # Compute DuDr
        autocorr = 0
        nthreads = 4  # Avoid nested parallelism
        bins = theta_bins

        DrDu = DDtheta_mocks(autocorr, nthreads, bins,
                             ra_galaxies, dec_galaxies,
                             RA2=ra_cluster_members, DEC2=dec_cluster_members,
                             link_in_dec=True, link_in_ra=True, verbose=False)

        # Compute DuRr
        RrDu = DDtheta_mocks(autocorr, nthreads, bins,
                             ra_randoms, dec_randoms,
                             RA2=ra_cluster_members, DEC2=dec_cluster_members,
                             link_in_dec=True, link_in_ra=True, verbose=False)

        Nr = float(len(ra_galaxies))
        Rr = float(len(ra_randoms))

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
                weight = dr_bins / rbins_cen
                wDuDr = np.sum(DrDu['npairs'] * weight)
                wDuRr = np.sum(RrDu['npairs'] * weight)                

        return wDuDr, wDuRr, Nr, Rr
    else:
        return 0.0, 0.0, 0.0, 0.0

        

def initialize_worker(shared_data):
    """
    Initialize worker processes by setting global variables.
    """
    import multiprocessing
    global global_shared_data
    global_shared_data = shared_data
    logging.info(f"Worker process {multiprocessing.current_process().name} initialized")



def run_wDuDr_wDuRr_jk_computation(npatches=100, h0=70.0, Om0=0.286, zmin_i=None, zmax_i=None):
    """
    Main function to compute wDuDr, wDuRr, Nr, Rr as a function of z_i, z_j, and jk_patch, and save the results.
    The Jackknife method is automatically chosen based on the number of unique clusters.

    Parameters:
    - npatches: number of Jackknife patches (default: 100)
    - h0: Hubble constant (default: 70.0 km/s/Mpc)
    - Om0: Matter density parameter (default: 0.286)
    - zmin_i: minimum redshift for z_i bins (default: 0.3)
    - zmax_i: maximum redshift for z_i bins (default: 0.35)
    """
    # Load data
    data = load_data(richness_min, richness_max)
    if data is None:
        logging.error("Data loading failed. Exiting.")
        return

    cluster_ids = data['cluster_ids']
    cluster_z_center = data['cluster_z_center']
    member_cluster_id = data['member_cluster_id']
    member_ra = data['member_ra']
    member_dec = data['member_dec']
    ra_ref = data['ra_ref']
    dec_ref = data['dec_ref']
    zspec_ref = data['zspec_ref']
    ra_ref_rand = data['ra_ref_rand']
    dec_ref_rand = data['dec_ref_rand']
    zspec_ref_rand = data['zspec_ref_rand']

    # Determine the Jackknife method based on the number of unique clusters
    num_clusters = len(cluster_ids)
    logging.info(f"Number of unique clusters: {num_clusters}")

    if num_clusters > 120:
        jk_method = 'spatial'
        logging.info("Using spatial Jackknife method (treecorr KMeans clustering).")
    else:
        jk_method = 'leave_one_cluster_out'
        logging.info("Using Leave-One-Cluster-Out Jackknife method.")

    # Define z_i bins (photometric bins)
    #dz_i = 0.0025
    z_i_bins = np.arange(zmin_i, zmax_i + dz_i, dz_i)
    n_zibins = len(z_i_bins) - 1

    # Digitize cluster_z_center into z_i bins
    z_i_tags = np.digitize(cluster_z_center, z_i_bins, right=True) - 1  # Adjust index to start from 0

    # Assign clusters to z_i bins
    clusters_by_zi = [[] for _ in range(n_zibins)]
    for idx, cid in enumerate(cluster_ids):
        z_i_tag = z_i_tags[idx]
        if 0 <= z_i_tag < n_zibins:
            # Get indices of members belonging to this cluster
            member_idx = np.where(member_cluster_id == cid)[0]
            cluster = {
                'id': cid,
                'member_idx': member_idx  # Store indices instead of boolean mask
            }
            clusters_by_zi[z_i_tag].append(cluster)

    jbins, jbins_zcen, jbins_dcen = micro_bins(zmin_j, zmax_j, dz_j, cosmo)
    # Eight linear bins between 0.1 and 10 Mpc
    rbins, dr_bins, rbins_cen = rbins_phydis(rmin, rmax, num_rbins, spacing=rbin_spacing)

    n_zjbins = len(jbins_zcen)
    n_rbins = len(rbins_cen)

    # Digitize zspec_reference into z_j bins
    z_j_tags = np.digitize(zspec_ref, jbins, right=True) - 1  # Adjust index to start from 0
    z_j_rand_tags = np.digitize(zspec_ref_rand, jbins, right=True) - 1  # Adjust index to start from 0

    # Assign galaxies to z_j bins
    galaxies_by_zj = []
    for j in range(n_zjbins):
        idx = z_j_tags == j
        galaxies = {
            'ra': ra_ref[idx],
            'dec': dec_ref[idx]
        }
        galaxies_by_zj.append(galaxies)

    # Assign randoms to z_j bins
    randoms_by_zj = []
    for j in range(n_zjbins):
        idx = z_j_rand_tags == j
        randoms = {
            'ra': ra_ref_rand[idx],
            'dec': dec_ref_rand[idx]
        }
        randoms_by_zj.append(randoms)

    # Prepare for computation
    num_processes = min(n_CPUs, cpu_count())
    logging.info(f"Using {num_processes} processes for computation")

    if jk_method == 'spatial':
        # Assign Jackknife labels using spatial patches
        ra_list = [member_ra, ra_ref, ra_ref_rand]
        dec_list = [member_dec, dec_ref, dec_ref_rand]
        jk_labels_list = assign_jk_labels(ra_list, dec_list, npatches)
        if jk_labels_list is None:
            logging.error("Jackknife label assignment failed.")
            return

        jk_labels_members  = jk_labels_list[0]
        jk_labels_ref      = jk_labels_list[1]
        jk_labels_ref_rand = jk_labels_list[2]

        # Prepare data structures
        max_jk = npatches
        wDuDr_array = np.zeros((n_zibins, n_zjbins, max_jk), dtype=np.float64)
        wDuRr_array = np.zeros((n_zibins, n_zjbins, max_jk), dtype=np.float64)
        Nr_array = np.zeros((n_zibins, n_zjbins, max_jk), dtype=np.float64)
        Rr_array = np.zeros((n_zibins, n_zjbins, max_jk), dtype=np.float64)
        jk_patch_labels = np.arange(max_jk)  # jk_patch labels from 0 to max_jk - 1

        def compute_wDuDr_wDuRr_spatial_jk(args):
            i, j, k_jk = args

            # Select cluster members in z_i bin
            clusters_in_bin = clusters_by_zi[i]
            if len(clusters_in_bin) == 0:
                return None

            member_indices = np.concatenate([cluster['member_idx'] for cluster in clusters_in_bin])
            cluster_members_ra = member_ra[member_indices]
            cluster_members_dec = member_dec[member_indices]
            cluster_members_jk_labels = jk_labels_members[member_indices]

            # Exclude data with JK label k_jk
            idx_members = cluster_members_jk_labels != k_jk
            cluster_members = {
                'ra': cluster_members_ra[idx_members],
                'dec': cluster_members_dec[idx_members]
            }

            # Select galaxies and randoms in z_j bin
            idx_zj = z_j_tags == j
            galaxies = {
                'ra': ra_ref[idx_zj],
                'dec': dec_ref[idx_zj]
            }
            galaxies_jk_labels = jk_labels_ref[idx_zj]

            idx_zj_rand = z_j_rand_tags == j
            randoms = {
                'ra': ra_ref_rand[idx_zj_rand],
                'dec': dec_ref_rand[idx_zj_rand]
            }
            randoms_jk_labels = jk_labels_ref_rand[idx_zj_rand]

            # Exclude data with JK label k_jk
            idx_galaxies = galaxies_jk_labels != k_jk
            galaxies = {
                'ra': galaxies['ra'][idx_galaxies],
                'dec': galaxies['dec'][idx_galaxies]
            }

            idx_randoms = randoms_jk_labels != k_jk
            randoms = {
                'ra': randoms['ra'][idx_randoms],
                'dec': randoms['dec'][idx_randoms]
            }

            if len(galaxies['ra']) == 0 or len(randoms['ra']) == 0 or len(cluster_members['ra']) == 0:
                return None

            # Compute angular diameter distance at z_j
            ang_diam_dist = jbins_dcen[j] / (1. + jbins_zcen[j])

            wDuDr, wDuRr, Nr, Rr = compute_wDuDr_wDuRr(cluster_members, galaxies, randoms,
                                                       rbins, dr_bins, rbins_cen, ang_diam_dist)
            return {'z_i': i, 'z_j': j, 'k_jk': k_jk,
                    'wDuDr': wDuDr, 'wDuRr': wDuRr, 'Nr': Nr, 'Rr': Rr}

        args_list = []
        for i in range(n_zibins):
            for j in range(n_zjbins):
                for k_jk in range(npatches):
                    args_list.append((i, j, k_jk))

        # Split args_list into chunks to reduce memory usage
        chunk_size = max(1, len(args_list) // num_processes)
        args_chunks = [args_list[k:k+chunk_size] for k in range(0, len(args_list), chunk_size)]

        # Parallel computation with progress bar
        logging.info("Starting parallel computation...")
        with Parallel(n_jobs=num_processes,
                      backend="loky") as pool:
            results = pool(delayed(compute_wDuDr_wDuRr_spatial_jk)(a)
                           for a in args_list)
            # Organize results into arrays
            for res in results:
                if res is not None:
                    i = res['z_i']
                    j = res['z_j']
                    k_jk = res['k_jk']
                    wDuDr_array[i, j, k_jk] += res['wDuDr']
                    wDuRr_array[i, j, k_jk] += res['wDuRr']
                    Nr_array[i, j, k_jk] += res['Nr']
                    Rr_array[i, j, k_jk] += res['Rr']

            # Clean up to free memory
            del results
            gc.collect()

        # Save the results
        filename = f'wDuDr_wDuRr_zj{dz_j}_zi{dz_i}_{jk_method}_{zmin_i}_{zmax_i}.h5'

        # Save the jk_patch labels
        jk_patch_labels = np.arange(npatches)

    elif jk_method == 'leave_one_cluster_out':
        # Prepare data structures
        total_clusters = len(cluster_ids)
        wDuDr_array = np.zeros((n_zibins, n_zjbins, total_clusters), dtype=np.float64)
        wDuRr_array = np.zeros((n_zibins, n_zjbins, total_clusters), dtype=np.float64)
        Nr_array = np.zeros((n_zibins, n_zjbins, total_clusters), dtype=np.float64)
        Rr_array = np.zeros((n_zibins, n_zjbins, total_clusters), dtype=np.float64)
        jk_patch_labels = cluster_ids  # jk_patch labels are cluster IDs

        # Map cluster IDs to indices
        cluster_idx_map = {cid: idx for idx, cid in enumerate(cluster_ids)}

        def compute_wDuDr_wDuRr_leave_one_cluster_out(args):
            i, j, idx_cluster = args
            clusters_in_bin = clusters_by_zi[i]
            if len(clusters_in_bin) == 0:
                return None

            # Exclude cluster idx_cluster
            clusters_excluded = [cluster for cluster in clusters_in_bin if cluster['id'] != idx_cluster]
            if not clusters_excluded:
                return None

            # Collect member galaxies of the remaining clusters
            member_indices = np.concatenate([cluster['member_idx'] for cluster in clusters_excluded])
            cluster_members = {
                'ra': member_ra[member_indices],
                'dec': member_dec[member_indices]
            }

            galaxies = galaxies_by_zj[j]
            randoms = randoms_by_zj[j]

            if len(galaxies['ra']) == 0 or len(randoms['ra']) == 0 or len(cluster_members['ra']) == 0:
                return None

            # Compute angular diameter distance at z_j
            ang_diam_dist = jbins_dcen[j] / (1. + jbins_zcen[j])

            wDuDr, wDuRr, Nr, Rr = compute_wDuDr_wDuRr(cluster_members, galaxies, randoms,
                                                       rbins, dr_bins, rbins_cen, ang_diam_dist)
            return {'z_i': i, 'z_j': j, 'idx_cluster': idx_cluster,
                    'wDuDr': wDuDr, 'wDuRr': wDuRr, 'Nr': Nr, 'Rr': Rr}

        args_list = []
        for i in range(n_zibins):
            clusters_in_bin = clusters_by_zi[i]
            cluster_ids_in_bin = [cluster['id'] for cluster in clusters_in_bin]
            for idx_cluster in cluster_ids_in_bin:
                for j in range(n_zjbins):
                    args_list.append((i, j, idx_cluster))

        # Split args_list into chunks to reduce memory usage
        chunk_size = max(1, len(args_list) // num_processes)
        args_chunks = [args_list[k:k+chunk_size] for k in range(0, len(args_list), chunk_size)]

        # Parallel computation with progress bar
        logging.info("Starting parallel computation...")
        with Parallel(n_jobs=num_processes,
                      backend="loky") as pool:
            results = pool(delayed(compute_wDuDr_wDuRr_leave_one_cluster_out)(a)
                           for a in args_list)

            # Organize results into arrays
            for res in results:
                if res is not None:
                    i = res['z_i']
                    j = res['z_j']
                    idx_cluster = res['idx_cluster']
                    idx = cluster_idx_map[idx_cluster]
                    wDuDr_array[i, j, idx] += res['wDuDr']
                    wDuRr_array[i, j, idx] += res['wDuRr']
                    Nr_array[i, j, idx] += res['Nr']
                    Rr_array[i, j, idx] += res['Rr']

            # Clean up to free memory
            del results
            gc.collect()

        # Save the results
        filename = f'wDuDr_wDuRr_zj{dz_j}_zi{dz_i}_{jk_method}_{zmin_i}_{zmax_i}.h5'

        # Save the jk_patch labels
        jk_patch_labels = cluster_ids

    else:
        logging.error("Invalid Jackknife method specified.")
        return

    # Save the results using HDF5
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset('rbins', data=rbins)
        hf.create_dataset('dr_bins', data=dr_bins)
        hf.create_dataset('rbins_cen', data=rbins_cen)
        hf.create_dataset('z_i_bins', data=z_i_bins)
        hf.create_dataset('z_j_bins', data=jbins)
        hf.create_dataset('wDuDr', data=wDuDr_array, compression='gzip')
        hf.create_dataset('wDuRr', data=wDuRr_array, compression='gzip')
        hf.create_dataset('Nr', data=Nr_array, compression='gzip')
        hf.create_dataset('Rr', data=Rr_array, compression='gzip')
        hf.create_dataset('jk_patch_labels', data=jk_patch_labels, compression='gzip')

    logging.info(f'wDuDr(z_i, z_j, jk_patch), wDuRr(z_i, z_j, jk_patch), Nr, Rr, and jk_patch_labels computed and saved using {jk_method} method.')



if __name__ == '__main__':
    # Start the timer
    start = timeit.default_timer()
    # Run the computation
    if len(zmin_i_arr) == 1:
            run_wDuDr_wDuRr_jk_computation(
            npatches=args_npatches,
            h0=args_h0,
            Om0=args_Om0,
            zmin_i=zmin_i_arr[0],
            zmax_i=zmax_i_arr[0])
    elif len(zmin_i_arr) >1 :
        for ii in range(len(zmin_i_arr)):
            run_wDuDr_wDuRr_jk_computation(
                npatches=args_npatches,
                h0=args_h0,
                Om0=args_Om0,
                zmin_i=zmin_i_arr[ii],
                zmax_i=zmax_i_arr[ii])

    else:
        logging.error(f"Please set wide tomo bin range in yml file.")

    # Stop the timer and print the elapsed time
    stop = timeit.default_timer()
    logging.info(f'Total time: {stop - start} seconds')

