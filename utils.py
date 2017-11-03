import cv2
import math
import numpy as np

def slope(p1, p2) :
    if p2[0] == p1[0]:
        return 0
    return (p2[1] - p1[1]) * 1. / (p2[0] - p1[0])

def y_intercept(slope, p1) :
   return p1[1] - 1. * slope * p1[0]

def intersect(line1, line2) :
   min_allowed = 1e-5   # guard against overflow
   big_value = 1e10     # use instead (if overflow would have occurred)
   m1 = slope(line1[0], line1[1])
   # print( 'm1: %d' % m1 )
   b1 = y_intercept(m1, line1[0])
   # print( 'b1: %d' % b1 )
   m2 = slope(line2[0], line2[1])
   # print( 'm2: %d' % m2 )
   b2 = y_intercept(m2, line2[0])
   # print( 'b2: %d' % b2 )
   if abs(m1 - m2) < min_allowed :
      x = big_value
   else :
      x = (b2 - b1) / (m1 - m2)
   y = m1 * x + b1
   y2 = m2 * x + b2
   # print( '(x,y,y2) = %d,%d,%d' % (x, y, y2))
   return (int(x),int(y))

def segment_intersect(line1, line2) :
   intersection_pt = intersect(line1, line2)

   # print( line1[0][0], line1[1][0], line2[0][0], line2[1][0], intersection_pt[0] )
   # print( line1[0][1], line1[1][1], line2[0][1], line2[1][1], intersection_pt[1] )

   if (line1[0][0] < line1[1][0]) :
      if intersection_pt[0] < line1[0][0] or intersection_pt[0] > line1[1][0] :
         # print( 'exit 1' )
         return None
   else :
      if intersection_pt[0] > line1[0][0] or intersection_pt[0] < line1[1][0] :
         # print( 'exit 2' )
         return None

   if (line2[0][0] < line2[1][0]) :
      if intersection_pt[0] < line2[0][0] or intersection_pt[0] > line2[1][0] :
         # print( 'exit 3' )
         return None
   else :
      if intersection_pt[0] > line2[0][0] or intersection_pt[0] < line2[1][0] :
         # print( 'exit 4' )
         return None

   return intersection_pt

def rect_intersect(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return (0,0,0,0)
    return (x, y, w, h)

def rect_contains(rec, point):
    return point[0] > rec[0] and point[0] < rec[0] + rec[2] and point[1] > rec[1] and point[1] < rec[1] + rec[3]

def rect_center(rec):
    return (int(rec[0] + rec[2] / 2),int(rec[1] + rec[3] / 2 ))

def rect_center_base(rec):
    return (int(rec[0] + rec[2] / 2),rec[1] + rec[3])

def rect_dist(rec1, rec2):
    cx1, cy1 = ((rec1[0] + rec1[2] )/ 2,(rec1[1] + rec1[3] )/ 2 )
    cx2, cy2 = ((rec2[0] + rec2[2] )/ 2,(rec2[1] + rec2[3] )/ 2 )

    return math.sqrt(math.pow(cx1 - cx2, 2) + math.pow(cy1 - cy2, 2))

def crop_image(image, rec, padding):
    return image[rec[1]-padding:rec[1]+rec[3]+padding, rec[0]-padding:rec[0]+rec[2]+padding]

def point_dist(p1, p2):
    xdist = p1[0] - p2[0]
    ydist = p1[1] - p2[1]
    return math.sqrt((xdist ** 2) + (ydist ** 2))

def create_hist(image):
    hist = cv2.calcHist([image],[0,1,2], None, [8,8,8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist,hist)

    return hist

def griddata(x, y, z, xdim, ydim, binsize=0.01, retbin=True, retloc=True):
    """
    Place unevenly spaced 2D data on a grid by 2D binning (nearest
    neighbor interpolation).

    Parameters
    ----------
    x : ndarray (1D)
        The idependent data x-axis of the grid.
    y : ndarray (1D)
        The idependent data y-axis of the grid.
    z : ndarray (1D)
        The dependent data in the form z = f(x,y).
    binsize : scalar, optional
        The full width and height of each bin on the grid.  If each
        bin is a cube, then this is the x and y dimension.  This is
        the step in both directions, x and y. Defaults to 0.01.
    retbin : boolean, optional
        Function returns `bins` variable (see below for description)
        if set to True.  Defaults to True.
    retloc : boolean, optional
        Function returns `wherebins` variable (see below for description)
        if set to True.  Defaults to True.

    Returns
    -------
    grid : ndarray (2D)
        The evenly gridded data.  The value of each cell is the median
        value of the contents of the bin.
    bins : ndarray (2D)
        A grid the same shape as `grid`, except the value of each cell
        is the number of points in that bin.  Returns only if
        `retbin` is set to True.
    wherebin : list (2D)
        A 2D list the same shape as `grid` and `bins` where each cell
        contains the indicies of `z` which contain the values stored
        in the particular bin.

    Revisions
    ---------
    2010-07-11  ccampo  Initial version
    """
    # get extrema values.
    xmin, xmax = xdim
    ymin, ymax = ydim

    # make coordinate arrays.
    xi      = np.arange(xmin, xmax+binsize, binsize)
    yi      = np.arange(ymin, ymax+binsize, binsize)
    xi, yi = np.meshgrid(xi,yi)

    # make the grid.
    grid           = np.zeros(xi.shape, dtype=x.dtype)
    nrow, ncol = grid.shape
    if retbin: bins = np.copy(grid)

    # create list in same shape as grid to store indices
    if retloc:
        wherebin = np.copy(grid)
        wherebin = wherebin.tolist()

    # fill in the grid.
    for row in range(nrow):
        for col in range(ncol):
            xc = xi[row, col]    # x coordinate.
            yc = yi[row, col]    # y coordinate.

            # find the position that xc and yc correspond to.
            posx = np.abs(x - xc)
            posy = np.abs(y - yc)
            ibin = np.logical_and(posx < binsize/2., posy < binsize/2.)
            ind  = np.where(ibin == True)[0]

            # fill the bin.
            bin = z[ibin]
            if retloc: wherebin[row][col] = ind
            if retbin: bins[row, col] = bin.size
            if bin.size != 0:
                binval         = np.median(bin)
                grid[row, col] = binval
            else:
                grid[row, col] = np.nan   # fill empty bins with nans.

    # return the grid
    if retbin:
        if retloc:
            return grid, bins, wherebin
        else:
            return grid, bins
    else:
        if retloc:
            return grid, wherebin
        else:
            return grid
