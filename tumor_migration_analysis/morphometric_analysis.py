#!/opt/local/bin/python

"""

This script contains several functions to extract morphometric data from cell segmentations.
It is intended to be used as a module to incorporate the functions into a migration analysis pipeline.
Running this script directly will generate some plots on the sample data.
The results are simply plotted, but lists of the data can easily be exported.

"""

import os
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import colors
from matplotlib import cm
import scipy.ndimage as ndi
from scipy.spatial import Voronoi
from skimage import color, morphology, draw, transform, measure
from skimage.io._plugins import tifffile_plugin as tifffile
from sklearn.neighbors import KDTree
from shapely.geometry import Point, Polygon, MultiPoint

def get_circular_se(radius=2):
    """Generates a 2D circular structuring element for binary operations

    Parameters
    ----------
    radius : int
         radius of the desired circular structuring element

    Returns
    -------
    se : 2D numpy array
         a 2D binary structuring element
    """

    n = (radius * 2) + 1
    se = np.zeros(shape=[n,n])
    for i in range(n):
        for j in range(n):
                se[i,j] = (i - np.floor(n / 2))**2 + (j - np.floor(n / 2))**2 <= radius**2
    se = np.array(se, dtype="uint8")
    return se

def polygonize_by_nearest_neighbor(pp):
    """Takes a set of xy coordinates pp Numpy array (n,2) and reorders the array to make
    a polygon using a nearest neighbor approach. Caution: this fails sometimes if there are
    sharp corners or "peninsulas" in the contour. You must flatten these out a bit.

    Parameters
    ----------
    pp : 2D numpy array
         an unordered array as (n,2) for (x,y) coordinates of an object contour

    Returns
    -------
    pp_new : 2D numpy array
         an ordered array as (n,2) for (x,y) coordinates of the input object contour
    """

    # start with first index
    pp_new = np.zeros_like(pp)
    pp_new[0] = pp[0]
    p_current_idx = 0

    tree = KDTree(pp)

    for i in range(len(pp) - 1):

        nearest_dist, nearest_idx = tree.query([pp[p_current_idx]], k=4)  # k1 = identity
        nearest_idx = nearest_idx[0]

        # finds next nearest point along the contour and adds it
        for min_idx in nearest_idx[1:]:  # skip the first point (will be zero for same pixel)
            if not pp[min_idx].tolist() in pp_new.tolist():  # make sure it's not already in the list
                pp_new[i + 1] = pp[min_idx]
                p_current_idx = min_idx
                break

    pp_new[-1] = pp[0]
    return pp_new

def voronoi_finite_polygons_2d(vor, radius=None):
    """Reconstruct infinite voronoi regions in a 2D diagram to finite regions.
    From Pauli Virtanen: https://gist.github.com/pv/8036995

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def extract_cell_areas(labels, px_size):
    """Returns the cell areas from a labeled image
    
    Parameters
    ----------
    labels : 2D numpy array
        a 2D labeled array (image) of the segmented regions
    px_size : float
        the pixel size (e.g. um/px)
    
    Returns
    -------
    areas : 1D numpy array
        an array of the the areas for each label
    """

    regionprops = measure.regionprops(labels)
    areas_px = np.array([_.area for _ in regionprops])
    areas_um = areas_px * px_size ** 2

    return areas_um

def angle_to_orientation(angle):
    """For an angle in radians from (-pi,pi), returns the orientation from (-pi/2,pi/2)

    Parameters
    ----------
    angle : float
        any angle in radians

    Returns
    -------
    angle : float
        the original angle given in radians from (-pi/2,pi/2)
    """

    angle = np.arctan2(np.sin(angle),np.cos(angle))
    if angle > np.pi/2:
        return angle - np.pi
    elif angle < -np.pi/2:
        return angle + np.pi
    else:
        return angle

def get_orientation_from_inertia(coords):
    """Calculates the principal axis of orientation for a set of coordinates
    using the eigenvalues of the moment of inertia. Also calculates the
    aspect ratio from the eigenvectors.

    Parameters
    ----------
    coords : 2D numpy array
        coordinate array in the form [[x1,x2,x3,xn],[y1,y2,y3,yn]]

    Returns
    -------
    theta : float
        the orientation of the principal axis in radians
    ar : float
        the aspect ratio of the coordinates (along the principal axis, always >=1)
    """

    #centers coordinates around centroid
    x0, y0 = coords.mean(axis=-1)
    x, y = coords
    x = x - x0
    y = y - y0

    #gets moments of inertia
    Ixx = (y ** 2).sum()
    Iyy = (x ** 2).sum()
    Ixy = (x * y).sum()
    I = np.array([[Ixx, -Ixy], [-Ixy, Iyy]])

    #gets eigenvalues and determines principal axis (theta)
    eigenvals, eigenvecs = np.linalg.eig(I)
    eigenvals = abs(eigenvals)
    loc = np.argsort(eigenvals)[::-1]
    d = eigenvecs[loc[0]]
    d *= np.sign(d[0])
    theta = np.arccos(d[1]) - np.pi/2
    theta = angle_to_orientation(theta)

    #gets aspect ratio
    eigenvals = eigenvals[loc]
    ar = np.sqrt(eigenvals[0]/eigenvals[1])

    return theta,ar

def get_orientation_ar(labels):
    """Returns the major axis orientation and aspect ratio for all labels in a labeled image

    Parameters
    ----------
    labels : 2D numpy array
         an image (usually 16bit) with the cell/object labels

    Returns
    -------
    orientation_list : 1D numpy array
        major axis of orientation for each label in radians from (-pi/2,pi/2) (order corresponds to the object labels starting with label 1 at index 0)
    ar_list : 1D numpy array
        aspect ratio values for each label (order corresponds to the object labels starting with label 1 at index 0)
    """

    orientation_list = []
    ar_list = []

    no_labels = np.max(labels)
    for i in range(1,no_labels+1): #label 1 is the border cells

        points = np.where(labels == i)
        points = np.array(points)

        theta, ar = get_orientation_from_inertia(points)
        orientation_list.append(theta)
        ar_list.append(ar)

    orientation_list = np.array(orientation_list)
    ar_list = np.array(ar_list)

    return orientation_list, ar_list

def get_sf(labels):
    """Returns the shape factor for each object in a labeled image

    Parameters
    ----------
    labels : 2D numpy array
         an image (usually 16bit) with the cell/object labels

    Returns
    -------
    sf_list : 1D numpy array
         shape factor for each label (order corresponds to the object labels starting with label 1 at index 0)
    """

    regionprops = measure.regionprops(labels)
    sf_list = np.array([_.perimeter / np.sqrt(_.area) for _ in regionprops])
    return sf_list

def isolate_largest_region(mask):

    mask = (mask > 0).astype('uint8')
    labels, no_labels = ndi.label(mask)
    sizes = ndi.sum(mask, labels, range(no_labels + 1))
    max_label = np.argmax(sizes[1:]) + 1
    return (labels == max_label).astype('uint8')

def polygon_from_mask(mask):

    #gets the outer contour
    mask = isolate_largest_region(mask)
    se = get_circular_se(radius=1)
    outer_mask = (ndi.binary_fill_holes(mask)).astype('uint8')
    contour = outer_mask - ndi.binary_erosion(outer_mask, structure=se, border_value=0)
    print(np.where(contour==1))
    pixels_mask = np.array(np.where(contour == 1)[::-1]).T
    outer = polygonize_by_nearest_neighbor(pixels_mask)

    #gets the inner contours
    inners = []
    holes = (ndi.binary_fill_holes(mask)).astype('uint8') - mask
    labels, no_labels = ndi.label(holes)
    for i in range(1,no_labels+1,1):
        inner_mask = labels == i
        inner_contour = inner_mask - (ndi.binary_erosion(inner_mask, structure=se)).astype('uint8')
        inner_pixels = np.array(np.where(inner_contour == 1)[::-1]).T
        inner = polygonize_by_nearest_neighbor(inner_pixels)
        if len(inner) > 3:
            inners.append(inner)

    # #plots to check
    # plt.imshow(contour)
    # plt.plot(pixels_mask[:,0],pixels_mask[:,1],'bo')
    # plt.plot(outer[:, 0], outer[:, 1], 'ro', ms=4)
    # for inner in inners:
    #     plt.plot(inner[:, 0], inner[:, 1], 'go', ms=4)
    # plt.show()
    # stop

    # makes a polygon from the outer and inner perimeters
    polygon = Polygon(outer,inners)

    return polygon

def tesselate_voronoi(labels,mask=None,px_size=1):
    """Does a Voronoi tesselation based on the centroids of the labeled image

    Parameters
    ----------
    labels : 2D numpy array
        an image (usually 16 bit) with the cell/object labels
    mask : 2D numpy array, optional
        an image (usually 8 bit binary) to show limit the tesselation region, in the  absence of a mask the whole frame is used
    px_size: float
        pixel size in units/px, optional

    Returns
    -------
    new_new_points : 2D numpy array
        label centroids in a [n,2] array as [x,y] (in units if px_size is specified)
    new_new_vertices : list of 2D numpy arrays
        vertices of the Voronoi polygons; each element is a [n,2] array as [x,y] (in units if px_size is specified)
    """

    #gets the centroids from the label image
    regionprops = measure.regionprops(labels)
    centroids = np.array([_.centroid[::-1] for _ in regionprops])

    #does a finite voronoi tesselation on the whole frame
    vor = Voronoi(centroids)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    # if no mask is specified, make a mask for the whole frame (with a 1px border)
    if mask is None:
        mask = np.ones_like(labels,dtype='uint8')
    else:
        mask = (mask > 0).astype('uint8')

    #generates a polygon from the mask
    polygon = polygon_from_mask(mask)

    # plots to check the mask polygon
    # fig,ax = plt.subplots()
    # ax.imshow(mask, cmap='Greys_r')
    # x, y = polygon.exterior.xy
    # ax.plot(x, y, c='b')
    # for interior in polygon.interiors:
    #     x, y = interior.xy
    #     ax.plot(x, y, c='r')
    # plt.show()
    # stop

    # clips tesselation to the mask polygon
    new_vertices = []
    for region in regions:
        poly_reg = vertices[region]
        shape = list(poly_reg.shape)
        shape[0] += 1
        p = Polygon(np.append(poly_reg, poly_reg[0]).reshape(*shape)).intersection(polygon)
        try:
            poly = np.array(list(zip(p.boundary.coords.xy[0][:-1], p.boundary.coords.xy[1][:-1])))
            new_vertices.append(poly * px_size)
        except (NotImplementedError, ValueError):  # sometimes there are multipolygons as a result - remove these
            print("No valid polygon for region", region)

    # reorders new points so that they are indexed to correspond with the regions (this is a bit slow)
    print("Reordering centroids and polygons")
    new_points = centroids * px_size
    new_points = np.vstack(_ for _ in new_points)
    new_new_points = np.array([0, 0])
    new_new_vertices = []
    for point in new_points:
        for vertex_list in new_vertices:
            if Polygon(vertex_list).contains(Point(point)):
                new_new_points = np.vstack((new_new_points, point))
                new_new_vertices.append(vertex_list)

    new_new_points = np.delete(new_new_points, 0, axis=0)

    return new_new_points, new_new_vertices

def get_centroids(labels):
    """Returns the cell centroids from a labeled image

    Parameters
    ----------
    labels : 2D numpy array
         an image (usually 16bit) with the cell/object labels

    Returns
    -------
    centriods : 2D numpy array
         label centroids as [x,y] in a [n,2] array
    """

    regionprops = measure.regionprops(labels)
    centroids = np.array([_.centroid[::-1] for _ in regionprops])
    return centroids

def test_voronoi():
    """Tests the Voronoi tesselation functions using some sample data"""

    data_dir = os.path.join('.','sample_data')
    px_size = 1

    #opens the original image
    img_name = 'nuclei.tif'
    img_path = os.path.join(data_dir,img_name)
    img = tifffile.imread(img_path)

    #makes a directory for saving
    plot_dir = os.path.join(data_dir,os.path.splitext(img_name)[0])
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)

    #opens the label image
    labels_name = 'nuclei_labeled.tif'
    labels_path = os.path.join(data_dir,labels_name)
    labels = tifffile.imread(labels_path)

    #does the Voronoi tesselation without a mask
    centroids, vertices = tesselate_voronoi(labels, px_size=px_size)

    #plots the result
    fig,ax = plt.subplots()
    ax.imshow(img, cmap='Greys_r', extent=(0, img.shape[1] * px_size, img.shape[0] * px_size, 0))
    for poly in vertices:
        ax.fill(*zip(*poly), alpha=0.5)
        ax.plot(*zip(*poly), ls='-', color='k', linewidth=0.5)
    ax.plot(centroids[:, 0], centroids[:, 1], 'ko', ms=3)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(plot_dir,'voronoi.pdf'))
    plt.close()

    #opens the mask and makes binary
    mask_name = 'mask.tif'
    mask_path = os.path.join(data_dir,mask_name)
    mask = tifffile.imread(mask_path)
    mask = (mask > 0).astype('uint8')

    #does the Voronoi tesselation with a mask
    centroids, vertices = tesselate_voronoi(labels, mask=mask, px_size=px_size)

    #plots the result
    fig,ax = plt.subplots()
    ax.imshow(img, cmap='Greys_r', extent=(0, img.shape[1] * px_size, img.shape[0] * px_size, 0))
    for poly in vertices:
        ax.fill(*zip(*poly), alpha=0.5)
        ax.plot(*zip(*poly), ls='-', color='k', linewidth=0.5)
    ax.plot(centroids[:, 0], centroids[:, 1], 'ko', ms=3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir,'voronoi_masked.pdf'))
    plt.close()

def test_ar_sf():
    """Uses some example data to test the functions for finding the aspect ratio and shape factor and makes some plots"""

    #gets the necessary file paths
    data_dir = os.path.join('.','sample_data')
    img_name = 'cell_ids.tif'
    img_path = os.path.join(data_dir,img_name)
    plot_dir = os.path.join(data_dir,os.path.splitext(img_name)[0])
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)

    #opens file and converts to grayscale
    print("Opening Image and Labeling")
    ids = tifffile.imread(img_path)
    ids = color.rgb2gray(ids)

    #relabels image
    bin = (ids > 0).astype('uint16')
    # optional: puts a border around the outside (this will then count the partial cells on the periphery)
    # bin[0,:] = 0
    # bin[-1,:] = 0
    # bin[:,0] = 0
    # bin[:,-1] = 0
    labels = morphology.label(bin,connectivity=1)

    #calculate the principal axis, aspect ratio and shape factor for each region
    print("Calculating morphometrics for all labels")
    no_labels = np.max(labels)

    centroid_list = get_centroids(labels)
    angle_list, ar_list = get_orientation_ar(labels)
    sf_list = get_sf(labels)

    # makes some plots
    print("Plotting")

    #makes a plot of just the binary outiline
    fig,ax = plt.subplots(figsize=(6,6))
    ax.imshow(bin,cmap='gray_r')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.subplots_adjust(bottom=0, left=0, top=1, right=1)
    plt.savefig(os.path.join(plot_dir, 'binary.pdf'))
    plt.close()

    #makes a plot of the labeled image
    fig,ax = plt.subplots(figsize=(6,6))
    np.random.seed(19680801)
    cmap = colors.ListedColormap(np.random.rand(256, 3))
    cmap.set_under('k')
    ax.imshow(labels,cmap=cmap,vmin=1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.subplots_adjust(bottom=0, left=0, top=1, right=1)
    plt.savefig(os.path.join(plot_dir, 'labels.pdf'))
    plt.close()

    #makes a plot of the binary with principal axes and ar
    fig,ax = plt.subplots(figsize=(6,6))
    ax.imshow(bin,cmap='gray_r')
    scale_factor = 0.25
    ax.quiver(centroid_list[1:,0], centroid_list[1:,1], ar_list[1:], ar_list[1:], units='xy', scale=scale_factor,
              scale_units='x', width=2, headlength=0, headaxislength=0, angles=angle_list[1:] * 180. / np.pi, color='r', pivot='middle')
    ax.plot(centroid_list[1:,0], centroid_list[1:,1],'go', ms=5)
    # for i in range(2,no_labels):
    #     ax.annotate("%i"%i,xy=centroid_list[i-1]+2,color='r',size=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.subplots_adjust(bottom=0, left=0, top=1, right=1)
    plt.savefig(os.path.join(plot_dir, 'aspect_ratio_quiver.pdf'))
    plt.close()

    #makes a plot of the aspect ratio scaled by color
    ar_img = deepcopy(labels)
    ar_img = ar_img.astype('float')
    ar_img[np.where(labels==1)] = 0

    for i in range(1,no_labels+1):
        points = np.where(labels == i)
        ar_img[points] = ar_list[i-1] #the ar list is from zero, labels are from 1

    fig,ax = plt.subplots(figsize=(6,6))
    cmap = plt.get_cmap('plasma')
    cmap.set_under('k')
    pos = ax.imshow(ar_img, cmap=cmap, vmin=1, vmax=2.6)
    cbar = fig.colorbar(pos, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks(np.arange(1,2.7,0.2))
    cbar.set_label('Aspect Ratio', rotation=90)
    ax.set_xticks([])
    ax.set_yticks([])
    # fig.subplots_adjust(bottom=0, left=0, top=1)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(plot_dir, 'aspect_ratio.pdf'))
    plt.close()

    #makes a plot of the shape factor scaled by color
    sf_img = deepcopy(labels)
    sf_img = sf_img.astype('float')
    sf_img[np.where(sf_img==1)] = 0

    for i in range(1,no_labels+1):
        points = np.where(labels == i)
        sf_img[points] = sf_list[i-1]  #the ar list is from zero, labels are from 1

    fig,ax = plt.subplots(figsize=(6,6))
    cmap = plt.get_cmap('plasma')
    cmap.set_under('k')
    pos = ax.imshow(sf_img, cmap=cmap, vmin=3.5, vmax=4.6)
    cbar = fig.colorbar(pos, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks(np.arange(3.5,4.7,0.1))
    cbar.set_label('Shape Factor', rotation=90)
    ax.set_xticks([])
    ax.set_yticks([])
    # fig.subplots_adjust(bottom=0, left=0, top=1)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'shape_factor.pdf'))
    plt.close()

def main():

    test_ar_sf()
    test_voronoi()

if __name__=="__main__":
    main()
