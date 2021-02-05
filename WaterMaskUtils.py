import numpy as np
from skimage import feature
from scipy.ndimage import gaussian_filter

def get_water_mask(eopatch, water_treshhold=0.4, canny_sigma=5, gauss_sigma=1):

    water_mask = eopatch.data['NDWI'].squeeze()
    water_mask = water_mask[:, :, :] >= water_treshhold
    water_mask = np.logical_or.reduce(water_mask)

    water_edges = feature.canny(water_mask, sigma=canny_sigma)

    water_area = ~gaussian_filter(~water_edges, sigma=gauss_sigma)

    return (water_mask, water_edges, water_area)

def visualise_water_mask():
    fig, ax = plt.subplots(2, 3, figsize=(16, 12))

    fig.suptitle('Lake Bilancino on {}'.format(eopatch.timestamp[0]))

    rgb = np.clip(eopatch.data['BANDS'][0][..., [band_names.index('B04'),band_names.index('B03'),band_names.index('B02')]] * 3, a_min=0, a_max=1)
    ax[0][0].set_title('RGB')
    ax[0][0].imshow(rgb, vmin=0, vmax=1, aspect='auto')

    pos = ax[0][1].imshow(eopatch.data['NDWI'][0].squeeze(), aspect='auto', vmin=-1, vmax=1)
    ax[0][1].set_title('NDWI')
    fig.colorbar(pos, ax=ax[0][1])

    lake_mask_treshold = 0.4
    lake_mask = eopatch.data['NDWI'].squeeze()
    lake_mask = lake_mask[:, :, :] >= lake_mask_treshold
    lake_mask = np.logical_or.reduce(lake_mask)
    ax[0][2].set_title('Lake maks as NDWI >= {}'.format(lake_mask_treshold))
    ax[0][2].imshow(lake_mask, aspect='auto')

    rgb2 = np.copy(rgb)
    rgb2[lake_mask] = 0
    ax[1][0].set_title('RGB with lake mask')
    ax[1][0].imshow(rgb2, aspect='auto')

    canny_sigma = 5
    lake_edges = feature.canny(lake_mask, sigma=canny_sigma)
    ax[1][1].set_title('Edge detection with Canny sigma {}'.format(canny_sigma))
    ax[1][1].imshow(lake_edges, aspect='auto')

    gaussian_filter_sigma = 1
    lake_edges_wide = ~gaussian_filter(~lake_edges, sigma=gaussian_filter_sigma)

    # lake_edges_wide = np.logical_and(~lake_mask, lake_edges_wide) # completly lake water out of mask

    ax[1][2].set_title('Expanded shores mask to get area of interest')
    ax[1][2].imshow(lake_edges_wide, aspect='auto')

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    rgb3 = np.copy(rgb)

    lake_edges_wide_borders = feature.canny(lake_edges_wide, sigma=canny_sigma)
    rgb3[lake_edges_wide_borders] = (1, 0, 0)
    ax[0].imshow(rgb3, aspect='auto')
    rgb3[lake_edges_wide] = (1, 0, 0)
    ax[0].imshow(rgb3, aspect='auto', alpha=0.1)
    ax[0].set_title('AoI over RGB')

    NDWI_AoI = np.copy(eopatch.data['NDWI'][0].squeeze())
    NDWI_AoI[~lake_edges_wide] = float('nan')

    ax[1].set_title('AoI over NDWI')
    ax[1].imshow(NDWI_AoI, aspect='auto', vmin=-1, vmax=1)