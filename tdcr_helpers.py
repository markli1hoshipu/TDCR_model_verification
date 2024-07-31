import numpy as np
import json
import draw_tdcr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import least_squares

# Code formatting has been adjusted under the help of ChatGPT

# Part 1 - Functions used for generating n16, for graphing visualization purposes

def generate_n16s(Rs, ps, seg_counts, total_lengths, model="foursegtdcr", filename='./mark_tdcr_curve_examples.json'):
    """
    Generate transformation matrices and save them to a JSON file.

    Parameters
    ----------
    Rs : list of np.ndarray
        List of rotation matrices.
    ps : list of np.ndarray
        List of position vectors.
    seg_counts : list of int
        List of segment counts for each transformation.
    total_lengths : list of float
        List of total lengths for each transformation.
    model : str, optional
        Model name, by default "foursegtdcr".
    filename : str, optional
        Filename to save the transformation matrices, by default './mark_tdcr_curve_examples.json'.
    
    Returns
    -------
    dict
        Dictionary containing the model name and transformation matrices.
    """
    re = {model: [np.eye(4)]}
    for idx in range(len(Rs)):
        scale = total_lengths[idx] / seg_counts[idx]
        length = np.linalg.norm(ps[idx])
        if is_SO3(Rs[idx]) and length > 0:
            ps[idx] = ps[idx].astype(np.float64)
            ps[idx] *= scale / length
            T = Rp_combine(Rs[idx], ps[idx])
            for _ in range(seg_counts[idx]):
                re[model].append(re[model][-1] @ T) 
        else:
            return False

    for idx in range(len(re[model])):
        re[model][idx] = convert_ndarray_to_list(flatten_col(re[model][idx]))

    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(re, json_file, ensure_ascii=False, indent=4)
    
    '''
    print('Segment count check:', all_elements_same(seg_counts))
    print('Lengths check:', all_elements_same(total_lengths))
    '''
    return re

def convert_ndarray_to_list(data):
    """
    Convert ndarray to list.

    Parameters
    ----------
    data : ndarray or other
        Input data to convert.
    
    Returns
    -------
    list or dict
        Converted data.
    """
    if isinstance(data, dict):
        return {k: convert_ndarray_to_list(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_ndarray_to_list(i) for i in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

def is_SO3(matrix):
    """
    Check if a matrix is a valid SO(3) rotation matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix to check.
    
    Returns
    -------
    bool
        True if the matrix is a valid SO(3) matrix, otherwise False.
    """
    if matrix.shape != (3, 3):
        return False
    identity = np.eye(3)
    if not np.allclose(matrix.T @ matrix, identity):
        return False
    if not np.isclose(np.linalg.det(matrix), 1):
        return False
    return True

def Rp_combine(R, p):
    """
    Combine rotation matrix and position vector into a transformation matrix.

    Parameters
    ----------
    R : np.ndarray
        Rotation matrix.
    p : np.ndarray
        Position vector.
    
    Returns
    -------
    np.ndarray
        Combined transformation matrix.
    """
    if R.shape != (3, 3):
        raise ValueError("R must be a 3x3 matrix")
    if p.shape == (3, 1) or p.shape == (3,):
        p = p.reshape(3, 1)  
    elif p.shape == (1, 3):
        p = p.T
    else:
        raise ValueError("p must be 3x1 or 1x3")
    
    return np.vstack((np.hstack((R, p)), np.array([[0, 0, 0, 1]])))

def flatten_col(matrix):
    """
    Flatten a matrix column-wise.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix to flatten.
    
    Returns
    -------
    np.ndarray
        Flattened matrix.
    """
    return matrix.reshape(-1, order='F')

def all_elements_same(lst):
    """
    Check if all elements in a list are the same.

    Parameters
    ----------
    lst : list
        Input list to check.
    
    Returns
    -------
    bool
        True if all elements are the same, otherwise False.
    """
    if not lst:
        return True  
    first_element = lst[0]
    for element in lst:
        if element != first_element:
            return False
    return True

def normalize(vector):
    """
    Normalize a vector.

    Parameters
    ----------
    vector : np.ndarray
        Input vector to normalize.
    
    Returns
    -------
    np.ndarray
        Normalized vector.
    """
    return vector / np.linalg.norm(vector)

def gen_rotation_matrix(axis, angle):
    """
    Generate a rotation matrix.

    Parameters
    ----------
    axis : list or np.ndarray
        Axis of rotation.
    angle : float
        Angle of rotation in radians.
    
    Returns
    -------
    np.ndarray
        Rotation matrix.
    """
    axis = normalize(np.array(axis))
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    one_minus_cos = 1 - cos_theta

    x, y, z = axis
    R = np.array([
        [cos_theta + x*x*one_minus_cos, x*y*one_minus_cos - z*sin_theta, x*z*one_minus_cos + y*sin_theta],
        [y*x*one_minus_cos + z*sin_theta, cos_theta + y*y*one_minus_cos, y*z*one_minus_cos - x*sin_theta],
        [z*x*one_minus_cos - y*sin_theta, z*y*one_minus_cos + x*sin_theta, cos_theta + z*z*one_minus_cos]
    ])
    
    return R

def test_part1_func1(n=10):
    """
    Test function for generating transformation matrices with single rotation.

    Parameters
    ----------
    n : int, optional
        Number of segments, by default 10.
        if n goes too big the program may get stuck
    """
    Rs = [gen_rotation_matrix(axis=[0, 1, 0], angle=np.pi / 2 / n)]
    ps = [np.array([[1/n, 0, 1]])]
    seg_counts = [n]
    total_lengths = [0.15]
    generate_n16s(Rs, ps, seg_counts, total_lengths)
    with open("./mark_tdcr_curve_examples.json", "r") as f:
        data = json.load(f)
    draw_tdcr.draw_tdcr_try(np.array(data["foursegtdcr"]), np.array([n]), 
                            tipframe=True, segframe=True, baseframe=True, projections=True, baseplate=True)

def test_part1_func2(n=10):
    """
    Test function for generating transformation matrices with multiple rotations.

    Parameters
    ----------
    n : int, optional
        Number of segments for each rotation, by default 10.
        if n goes too big the program may get stuck
    """
    Rs = [
        gen_rotation_matrix(axis=[0, 1, 0], angle=np.pi / 8 / n),
        gen_rotation_matrix(axis=[1, 0, 0], angle=-np.pi / 3 / n),
        gen_rotation_matrix(axis=[0, 0, 1], angle=np.pi / 6 / n)
    ]
    ps = [
        np.array([[1, 0, 1]]),
        np.array([[0, 2, 3]]),
        np.array([[1, 2, 0]])
    ]
    seg_counts = [n] * 3
    total_lengths = [0.2] * 3
    generate_n16s(Rs, ps, seg_counts, total_lengths)
    with open("./mark_tdcr_curve_examples.json", "r") as f:
        data = json.load(f)
    draw_tdcr.draw_tdcr_try(np.array(data["foursegtdcr"]), np.array([n, n*2, n*3]), 
                            tipframe=True, segframe=True, baseframe=True, projections=True, baseplate=True)



# test_part1_func1(n = 10)
# result: it seems the basic functionalities are working, but input parameters can be refined.

# Part 3 - Real Data Fitting

def plot_points(list_of_vectors):
    """
    Plot points in 3D space and return the Axes3D object.

    Parameters
    ----------
    list_of_vectors : list
        A list where each element is a 1x3 numpy array or list representing a point in 3D space.
    
    Returns
    -------
    Axes3D
        Axes3D object for the plot.
    """
    points = np.vstack(list_of_vectors)

    fig = plt.figure()
    fig.set_size_inches(1280 / fig.dpi, 1024 / fig.dpi)
    ax = fig.add_subplot(projection='3d', computed_zorder=False)

    clearance = 10
    min_val_x, max_val_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_val_y, max_val_y = np.min(points[:, 1]), np.max(points[:, 1])
    min_val_z, max_val_z = np.min(points[:, 2]), np.max(points[:, 2])
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(min_val_x - clearance, max_val_x + clearance)
    ax.set_ylim(min_val_y - clearance, max_val_y + clearance)
    ax.set_zlim(min_val_z - clearance, max_val_z + clearance)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.grid(True, alpha=0.3)
    ax.view_init(azim=45, elev=30)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')

    return ax

def calc_R(xc, yc, zc, points):
    """
    Calculate the distance from points to the center (xc, yc, zc).

    Parameters
    ----------
    xc, yc, zc : float
        Coordinates of the center.
    points : np.ndarray
        Points to calculate the distance.
    
    Returns
    -------
    np.ndarray
        Distances from the points to the center.
    """
    return np.sqrt((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2 + (points[:, 2] - zc)**2)

def f_3(c, points):
    """
    Objective function for optimization: error of the distances from points to the circle center.

    Parameters
    ----------
    c : list or np.ndarray
        Coordinates of the circle center.
    points : np.ndarray
        Points to calculate the error.
    
    Returns
    -------
    np.ndarray
        Error of the distances.
    """
    Ri = calc_R(*c, points)
    return 2 * Ri - Ri.mean()

def fit_sphere(points):
    """
    Fit a sphere to a set of points.

    Parameters
    ----------
    points : np.ndarray
        Points to fit the sphere.
    
    Returns
    -------
    tuple
        Coordinates of the center and the radius of the sphere.
    """
    center_estimate = np.mean(points, axis=0)
    center_optimized = least_squares(f_3, center_estimate, args=(points,))
    xc, yc, zc = center_optimized.x
    radius = np.mean(calc_R(xc, yc, zc, points))
    return xc, yc, zc, radius

def plot_points_and_circle_center(list_of_vectors):
    """
    Plot points in 3D space and compute the circle center.

    Parameters
    ----------
    list_of_vectors : list
        A list where each element is a 1x3 numpy array or list representing a point in 3D space.
    
    Returns
    -------
    Axes3D
        Axes3D object for the plot.
    """
    ax = plot_points(list_of_vectors)
    points = np.vstack(list_of_vectors)
    xc, yc, zc, radius = fit_sphere(points)
    ax.scatter(xc, yc, zc, c='b', marker='o', s=100)
    plt.show()
    return ax

def plot_disks(list_of_listofcircum):
    """
    Plot disks in 3D space.

    Parameters
    ----------
    list_of_listofcircum : list
        List of lists where each inner list contains 1x3 vectors representing a disk.
    
    Returns
    -------
    Axes3D
        Axes3D object for the plot.
    """
    for list_of_vectors in list_of_listofcircum:
        ax = plot_points(list_of_vectors)
        points = np.vstack(list_of_vectors)
        xc, yc, zc, radius = fit_sphere(points)
        ax.scatter(xc, yc, zc, c='b', marker='o', s=100)
    plt.show()
    return ax

'''
# Example usage:
list_of_vectors = [\
    [-82.2108, -301.8995, 98.2890],
    [-85.9516, -296.9931, 97.6929],
    [-88.8214, -295.2875, 97.4193],
    [-97.4022, -295.5168, 97.8696],
    [-102.8964, -300.5965, 97.5369],
    [-103.9620, -303.7377, 97.8663],
    [-87.9619, -316.0418, 99.4617],
    [-83.1042, -310.8223, 98.4542],
    [-84.9767, -313.4016, 99.1352],
    [-97.5344, -315.9448, 99.4549],
    [-98.3666, -315.7011, 99.5154],
    [-101.5966, -312.8887, 98.6481],
    [-82.2031, -301.7092, 97.8819],
    [-87.3057, -296.4393, 97.5356],
    [-83.5214, -300.2137, 97.6650]
]

plot_points_and_circle_center(list_of_vectors)
'''