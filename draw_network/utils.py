import numpy as np
import math
import logging


def calculate_angle_from_slope(dx, dy):
    """
    Calculates the angle in degrees from a slope represented by rise and run.

    Args:
      dy: The vertical change of the slope.
      dx: The horizontal change of the slope.

    Returns:
      The angle of the slope in degrees.
    """

    if dx == 0:
    # Handle the case where the slope is vertical
        return 90 if dy > 0 else -90
    else:
        slope = dy / dx
        angle_in_radians = math.atan(slope)
        angle_in_degrees = math.degrees(angle_in_radians)
        return angle_in_degrees

def is_straight_line(points):
    """
    This function checks if a list of points represents a straight line.

    Args:
      points: A list of 2D points, where each point is a tuple (x, y).

    Returns:
      True if the points are on a straight line, False otherwise.
    """
    if len(points) < 3:
        return True  # Less than 3 points can always be considered a line

    # Check for all points having the same x-coordinate (vertical line)
    all_same_x = True
    x_ref = points[0][0]  # Reference x-coordinate
    for x, _ in points[1:]:
        if x != x_ref:
            all_same_x = False
            break

    if all_same_x:
        return True  # All points have the same x, hence a vertical line

    # Calculate the slope for non-vertical lines (avoid division by zero)
    x1, y1 = points[0]
    x2, y2 = points[1]
    slope = (y2 - y1) / (x2 - x1) if x1 != x2 else float('inf')  # Set slope to infinity for vertical line

    # Check if remaining points follow the same slope (or are on the same vertical line)
    for x, y in points[2:]:
        if all_same_x:
            # For vertical line, check if points have the same x-coordinate
            if x != x_ref:
                return False
        else:
            # For non-vertical line, check slope within tolerance
            if abs((y - y1) - slope * (x - x1)) > 1e-6:
                return False

    return True

def get_missing_rotated_rectangle_points(point1, point2, angle):
    """
    this is a wrapper for get_missing_rectangle_points, but some rotation and backrotation are performed before and after.
    """

    rot_P1 = rotate_point(point1,(0,0),-angle)
    rot_P2 = rotate_point(point2,(0,0),-angle)
    R1, R2 = get_missing_rectangle_points(rot_P1, rot_P2)
    bR1 = rotate_point(R1,(0,0),angle)
    bR2 = rotate_point(R2,(0,0),angle)
    return bR1, bR2

def get_missing_rectangle_points(point1, point2):
    """
    Calculates the other two points to construct a rectangle given two diagonal points.

    Args:
      point1: A numpy array of size 2 representing the first point (x1, y1).
      point2: A numpy array of size 2 representing the second point (x2, y2).

    Returns:
        missing reactangle points.
    """
    point1 = np.array(point1)
    point2 = np.array(point2)

    diff = point2 - point1

    # Determine the dominant axis based on the absolute difference
    dominant_axis = np.argmax(np.abs(diff))  # 0 for x-axis, 1 for y-axis

    # Define corner positions based on dominant axis and point order
    if dominant_axis == 0:  # x-axis is dominant
        if point1[0] < point2[0]:  # point1 is leftmost
            bottom_left = point1
            top_right = point2
        else:  # point2 is leftmost
            bottom_left = point2
            top_right = point1
    else:  # y-axis is dominant
        if point1[1] < point2[1]:  # point1 is lower
            bottom_left = point1
            top_right = point2
        else:  # point2 is lower
            bottom_left = point2
            top_right = point1

    # Calculate the remaining two corners based on the bottom-left and top-right
    top_left = np.array([bottom_left[0], top_right[1]])
    bottom_right = np.array([top_right[0], bottom_left[1]])

    # Return all four corners in clockwise order
    return np.array([bottom_right, top_left])

def line_circle_intersection(line_p1, line_p2, circle_center, circle_radius):
    """
    Finds the intersection points of a line and a circle.

    Args:
      line_p1: A numpy array of size 2 representing the first point of the line.
      line_p2: A numpy array of size 2 representing the second point of the line.
      circle_center: A numpy array of size 2 representing the center of the circle.
      circle_radius: The radius of the circle.

    Returns:
      A numpy array of size 2 x 2 containing the intersection points,
      or None if there are no intersections.
    """

    # Line direction vector
    d = np.array(line_p2) - np.array(line_p1)

    # Vector from circle center to line p1
    s = np.array(line_p1) - np.array(circle_center)

    # Projection of s onto d
    a = np.dot(s, d) / np.dot(d, d)

    # Closest point on line to circle center
    proj = line_p1 + a * d

    # Distance between closest point and circle center
    dist = np.linalg.norm(proj - circle_center)

    # Check if intersection is possible
    if dist > circle_radius:
        return None

    # Distance from projection to intersection point (along d)
    h = np.sqrt(circle_radius**2 - dist**2)

    # Intersection points
    intersection1 = proj + a * d + h * d / np.linalg.norm(d)
    intersection2 = proj + a * d - h * d / np.linalg.norm(d)

    return np.array([intersection1, intersection2])

def bezier_curve(bez_points):
    """
    Calculates a point on a Bézier curve using the de Casteljau algorithm.
    """
    if is_straight_line(bez_points):
        return bez_points.T
    else:
        theta = np.linspace(0,1,101)
        n = len(bez_points) - 1

        def bernstein_basis(i, n, t):
            return np.math.factorial(n) / (np.math.factorial(i) * np.math.factorial(n - i)) * t**i * (1 - t)**(n - i)

        def calculate_bezier_curve_point(bez_points,t):
            P = np.zeros_like(bez_points[0])
            for i in range(n + 1):
                P += bernstein_basis(i, n, t) * bez_points[i]
            return P

        points = np.array([calculate_bezier_curve_point(bez_points, t) for t in theta]).T

        return points

def rotate_point(point1, center, angle=90):
    """
    TODO
    """

    # Convert angle to radians
    angle_rad = np.radians(angle)

    # Rotation matrix for 90 degrees counter-clockwise
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])

    # Translate points to center, rotate, translate back
    rotated_point1 = center + rotation_matrix.dot(point1 - center)

    return rotated_point1

def rotate_points_over_center(point1,point2,angle=90):
    center = np.array([np.mean([point1[0],point2[0]]),np.mean([point1[1],point2[1]])])
    R1 = rotate_point(point1,center,angle)
    R2 = rotate_point(point2,center,angle)
    return R1, R2

def get_closer_point(P1,P2,center):
    """
    return either P1, P2, whichever is closer to the center point.
    """
    norm1 = np.linalg.norm(P1-center)
    norm2 = np.linalg.norm(P2-center)
    if norm1 > norm2:
        return P2
    else:
        return P1

def separate_arrow(arrow):
    """
    Separates arrow for a single edge into a list of arrows for two edges.
    """

    if arrow == "<-":
        arrow = ["<-","-"]
    elif arrow == "->":
        arrow = ["-","->"]
    elif arrow == "<->":
        arrow = ["<-","->"]
    elif arrow == "-":
        arrow = ["-","-"]
    else:
        logging.warning(f"Arrow '{arrow}' not implemented, default to '-'.")
        arrow = ["-","-"]
    return arrow

def separate_dict_of_lists_into_list_of_dicts(arg_list_len,edge_kwargs={},node_kwargs={},node_text_kwargs={},text=None):
    #--- SEPARATE ALL THE DICTS OF LISTS INTO A LIST OF DICTS ---#
    node_text_kwargs = node_text_kwargs.copy()
    edge_kwargs = edge_kwargs.copy()
    node_kwargs = node_kwargs.copy()

    if "plot_kwargs" not in edge_kwargs.keys():
        edge_kwargs["plot_kwargs"] = {}
    edge_kwargs["plot_kwargs"] = separate_list_kwarg_dicts(edge_kwargs["plot_kwargs"],"edgeplot_",[],arg_list_len)
    edge_kwargs = separate_list_kwarg_dicts(edge_kwargs,None,[],arg_list_len)

    if "plot_kwargs" not in node_kwargs.keys():
        node_kwargs["plot_kwargs"] = {}
    node_kwargs["text_kwargs"] = separate_list_kwarg_dicts(node_text_kwargs,"nodetext_",[],arg_list_len)
    node_kwargs["plot_kwargs"] = separate_list_kwarg_dicts(node_kwargs["plot_kwargs"],"nodeplot_",[],arg_list_len)
    node_kwargs = separate_list_kwarg_dicts(node_kwargs,None,[],arg_list_len)
    return edge_kwargs, node_kwargs

def line_intersection(p1, p2, p3, p4):
    """
    Find the intersection point of two line segments (p1, p2) and (p3, p4).

    Parameters:
    p1, p2, p3, p4 : tuple
        Tuples representing the coordinates of the endpoints of the two line segments.

    Returns:
    tuple or None
        The intersection point (x, y) if the lines intersect within the segments, otherwise None.
    """
    def on_segment(p, q, r):
        if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
            return True
        return False

    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        elif val > 0:
            return 1
        else:
            return 2

    def do_intersect(p1, q1, p2, q2):
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        if o1 != o2 and o3 != o4:
            return True
        if o1 == 0 and on_segment(p1, p2, q1):
            return True
        if o2 == 0 and on_segment(p1, q2, q1):
            return True
        if o3 == 0 and on_segment(p2, p1, q2):
            return True
        if o4 == 0 and on_segment(p2, q1, q2):
            return True
        return False

    def intersection_point(p1, p2, p3, p4):
        A1 = p2[1] - p1[1]
        B1 = p1[0] - p2[0]
        C1 = A1 * p1[0] + B1 * p1[1]

        A2 = p4[1] - p3[1]
        B2 = p3[0] - p4[0]
        C2 = A2 * p3[0] + B2 * p3[1]

        determinant = A1 * B2 - A2 * B1

        if determinant == 0:
            return None
        else:
            x = (B2 * C1 - B1 * C2) / determinant
            y = (A1 * C2 - A2 * C1) / determinant
            return (x, y)

    if do_intersect(p1, p2, p3, p4):
        return intersection_point(p1, p2, p3, p4)
    else:
        return None

def get_intersections(node_points, line_point1, line_point2):
    """
    Calculates intersection points between the line and an approximation of the node using line segments between consecutive points.

    Parameters:
    center : tuple of floats
      The (x, y) coordinates of the node's center.
    radius : float
      The radius of the node circle.
    line_point1 : tuple of floats
      The (x, y) coordinates of the first point on the line.
    line_point2 : tuple of floats
      The (x, y) coordinates of the second point on the line.
    width : float, optional
      The width of the rectangular node shape (default: 0, circular).
    height : float, optional
      The height of the rectangular node shape (default: 0, circular).

    Returns:
    list of tuples
      A list containing tuples of (x, y) coordinates for each intersection point.
    """

    # List to store intersections
    intersections = []

    # Iterate through consecutive node points
    for i in range(len(node_points[0]) - 1):
    # Extract current and next point
        point1 = (node_points[0][i], node_points[1][i])
        point2 = (node_points[0][i + 1], node_points[1][i + 1])

        # Find intersection between line segment and line
        local_intersections = line_intersection(point1, point2, line_point1, line_point2)


        if local_intersections is not None:
            intersections.append(local_intersections)

    if len(intersections) == 0:
        import matplotlib.pyplot as plt
        plt.plot(*node_points,zorder=100)

    return intersections

def get_node_coordinates(center,radius,width=0,heigth=0):
    """
    Generates polar coordinates for a circle representing a node.

    Parameters
    ----------
    center : tuple of floats
        The (x, y) coordinates of the node's center.
    radius : float
        The radius of the node circle.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing two NumPy arrays representing the x and y coordinates of the node circle.
    """

    #--- CALCULATE POINTS ON CIRCLE ---#
    n = 100
    n4 = n//4

    theta = np.linspace(0, 2 * np.pi, n)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)


    #--- APPLY WIDTH AND HEIGHT ---#
    top_left = np.arange(n4*1,n4*2)
    bot_left = np.arange(n4*2,n4*3)
    top_righ = np.arange(n4*0,n4*1)
    bot_righ = np.arange(n4*3,n4*4)

    x[np.append(top_left,bot_left)] -= width
    x[np.append(top_righ,bot_righ)] += width
    y[np.append(top_left,top_righ)] += heigth
    y[np.append(bot_left,bot_righ)] -= heigth

    x = np.append(x,x[0])
    y = np.append(y,y[0])
    return np.array([x, y])

def separate_list_kwarg_dicts(kwargs_dict,keyword,exclusion_list,list_len):
    from .core import default

    default_parameters = default.parameters
    kwargs_dict = kwargs_dict.copy()

    kwargs_dicts = []
    for i in range(list_len):
        # update parsed kwargs with default parameters
        if keyword is not None:
            for key, item in default_parameters.items():
                if keyword == key[:len(keyword)]:
                    passes_as = key.replace(keyword,"")
                    if passes_as not in kwargs_dict.keys() and passes_as not in exclusion_list:
                        kwargs_dict[passes_as] = item
        # separate kwarg dicts
        tmp_kwargs_dict = {}
        for key, item in kwargs_dict.items():
            if type(item) == list:
                assert len(item) == list_len, f"Incorrect length of kwargs list. Expected {list_len}, got {len(item)}."
                tmp_kwargs_dict[key] = item[i]
            else:
                tmp_kwargs_dict[key] = item
        kwargs_dicts.append(tmp_kwargs_dict)

    return kwargs_dicts

def is_point_inside_convex_polygon(x_coords, y_coords, point):
    def cross_product(x1, y1, x2, y2, x3, y3):
        # Returns the cross product of vectors (x2 - x1, y2 - y1) and (x3 - x1, y3 - y1)
        # A positive cross product indicates a counter-clockwise turn
        # A negative cross product indicates a clockwise turn
        # A zero cross product indicates the points are collinear
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

    n = len(x_coords)
    inside = True
    px, py = point

    for i in range(n):
        # Get the vertices of the edge
        x1, y1 = x_coords[i], y_coords[i]
        x2, y2 = x_coords[(i + 1) % n], y_coords[(i + 1) % n]

        # Check the orientation of the point with respect to the edge
        if cross_product(x1, y1, x2, y2, px, py) < 0:
            inside = False
            break

    return inside

def calculate_single_rotated_bezier_point(center_N1,center_N2,value):
    rotated_N1, rotated_N2 = rotate_points_over_center(center_N1,center_N2)
    correct_value = (value/2+0.5)
    # single_symmetric_bezier_point
    srbp = rotated_N1*(correct_value)+rotated_N2*(1-correct_value)
    return srbp
