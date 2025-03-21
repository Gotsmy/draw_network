import numpy as np
import matplotlib.pyplot as plt
import math
from . import utils
import logging

class draw_network_default_parameters:
    """
    Stores  all the default parameters. The parameter names are separated into a
    keyword and an argument by the *initial* underscore. Keywords indicate which
    function a the parameter is parsed to. The argument indicates which argument
    the parameter stands for.Following keywords are implemented:

    * "canvas" indicates arguments parsed to create_canvas().
    * "node" indicates arguments parsed to create_node().
    * "edge" indicates arguments parsed to create_edge().
    * "nodeplot" indicates arguments parsed to plt.fill_between() over create_node(plot_kwargs).
    * "edgeplot" indicates arguments parsed to plt.plot() over create_edge(plot_kwargs).
    * "nodetext" indicates arguments parsed to plt.text() over create_node(text_kwargs).
    * "edgetext" indicates arguments parsed to plt.text() over create_edge(text_kwargs).
    * "coreactant" indicates arguments parsed to add_coreactants().

    """

    parameters = {
        "node_radius": .5,
        "node_width": 0,
        "node_height": 0,
        "node_fill_color": "white",
        #---
        "nodeplot_linewidth": 1.5,
        "nodeplot_color": "black",
        "nodeplot_zorder": 2,
        #---
        "nodetext_fontsize": 20,
        "nodetext_va": "center",
        "nodetext_ha": "center",
        "nodetext_zorder": 4,
        #---
        "edgeplot_color": "black",
        "edgeplot_solid_capstyle": "round",
        "edgeplot_linewidth": 1.5,
        #---
        "edge_offset": 0.015,
        "edge_curve": 0.,
        "edge_bezier_calculation": "rectangle", # "rotation"
        "edge_line": 0.,
        "edge_symmetry": False,
        "edge_arrow": "-",
        "edge_arrow_angle": 30,
        "edge_arrow_length": .15,
        "edge_rectangle_angle": 0,
        "edge_text_offset": (0,0),
        "edge_debug": False, # This parameter can only be set here. If True, then Bezier Points are plottet.
        #---
        "edgetext_fontsize": 20,
        "edgetext_va": "center",
        "edgetext_ha": "left",
        "edgetext_zorder": 4,
        #---
        "coreactant_side": "both",
        "coreactant_arrow": "-",
        "coreactant_width": .0,
        "coreactant_height": .2,
        "coreactant_bez_width": None,
        #---
        "canvas_dpi": 200,
    }

    def __init__(self):
        pass

    def update_parameter(self,keyword,parameter):
        """
        Function to update a single parameter.

        Input
        -----
        keyword : str
            The keyword must contain the full keyword_argument name as explained in this class' docstring. Do "print(draw_network.default.parameters)" to see which parameters can be set.
        parameter : obj
            Correct parameter data type. Please refer to the relevant function which the parameter is parsed to as explained in this class' docstring.
        """
        assert keyword in self.parameters.keys(), f"Keyword {keyword} is not found. Nothing is updated."
        self.parameters[keyword] = parameter

    def update_parameters(self,keywords,parameters):
        """
        Function to update multiple parameters.

        Input
        -----
        keywords : list of str
            List of keywords. The keyword must contain the full keyword_argument name as explained in this class' docstring. Do "print(draw_network.default.parameters)" to see which parameters can be set.
        parameter : list of obj
            List of correct parameter data types. Please refer to the relevant function which the parameter is parsed to as explained in this class' docstring.
        """
        assert len(keywords)==len(parameters)
        for keyword, parameter in zip(keywords,parameters):
            self.update_parameter(keyword,parameter)

default = draw_network_default_parameters()

def create_canvas(xlim,ylim,dpi=None):
    """
    Calculated (node) circles are just displayed as circles and not ellipses if
    the  figure  size  fits  to  the  set xlim and ylim arguments. This function
    calculates  the  figure size and creates a Matplotlib figure and axis object
    with the correct aspect ratio.

    Parameters
    ----------
    xlim : tuple of floats
        The x-axis limits of the canvas.
    ylim : tuple of floats
        The y-axis limits of the canvas.
    dpi : int, optional
        The dots-per-inch resolution of the figure. If None, default parameter is used.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure object.
    ax : matplotlib.axes.Axes
        Matplotlib axis object.

    Raises
    ------
    AssertionError
        If the provided x-axis or y-axis limits are negative or zero.
    """
    #--- LOAD DEFAULT PARAMETERS & UPDATE KWARGS ---#
    dpi = dpi if dpi is not None else default.parameters["canvas_dpi"]

    deltaX = xlim[1] - xlim[0]
    assert deltaX > 0, "Negative or 0 canvas size in X-axis."

    deltaY = ylim[1] - ylim[0]
    assert deltaY > 0, "Negative or 0 canvas size in Y-axis."



    fig = plt.figure(dpi=dpi, figsize=(deltaX, deltaY))
    ax = plt.subplot(111)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax

def create_node(center,ax=None,radius=None,text=None,width=None,height=None,fill_color=None,plot_kwargs={},text_kwargs={}):
    # TODO: make width, heigth, fill_color default parameters settable in the default dict.
    """
    Creates a dictionary containing information about a node for network visualization.

    Parameters
    ----------
    center : tuple of floats
        The (x, y) coordinates of the node's center.
    ax : plt.Axes object, optional
        The matplotlib Axes object to plot the node on. Defaults to None.
    radius : float, optional
        The radius of the node circle. If not provided, the default radius from parameters is used.
    text : str, optional
        The label text to display within the node. Defaults to None.
    width : float >= 0
        Stretches the node horizontally
    height : float >= 0
        Stretches the node vertically
    fill_color : str
        Matplotlib compatible color string. Will be the background color of the node.
    plot_kwargs : dict, optional
        A dictionary of keyword arguments for customizing the node circle's plot style.
        Defaults to an empty dictionary.
    text_kwargs : dict, optional
        A dictionary of keyword arguments for customizing the node's label text style.
        Defaults to an empty dictionary.

    Returns
    -------
    dict
        A dictionary containing information about the node, including its center, radius, circle
        coordinates, plot style arguments, label text (if provided), and label text style arguments.
    """
    #--- LOAD DEFAULT PARAMETERS & UPDATE KWARGS ---#
    if radius is None:
        radius = default.parameters["node_radius"]
    if width is None:
        width = default.parameters["node_width"]
    if height is None:
        height = default.parameters["node_height"]
    if fill_color is None:
        fill_color = default.parameters["node_fill_color"]
    plot_kwargs, = utils.separate_list_kwarg_dicts(plot_kwargs,"nodeplot_",[],1)
    text_kwargs, = utils.separate_list_kwarg_dicts(text_kwargs,"nodetext_",[],1)
    #-------------------------------#


    node_dict = {
        "center": np.array(center),
        "radius": radius,
        "points": utils.get_node_coordinates(center,radius,width,height),
        "plot_kwargs": plot_kwargs,
        "text": text,
        "text_kwargs": text_kwargs,
        "width": width,
        "height": height,
        "fill_color": fill_color
    }

    if ax is not None:
        draw_node(node_dict,ax)
    return node_dict


def draw_node(node_dict,ax):
    """
    Plots a node or edge and its label text (if provided) on a matplotlib axis.

    Parameters
    ----------
    node_dict : dict
        A dictionary containing information about the node, as generated by `create_node`.

    ax : plt.Axes object
        The matplotlib Axes object to plot the node on.
    """

    outer = node_dict["points"]
    inner_radius = node_dict["radius"]-node_dict["plot_kwargs"]["linewidth"]/60
    if inner_radius < 0:
        inner_radius = node_dict["radius"]
        inner_width = node_dict["width"]-node_dict["plot_kwargs"]["linewidth"]/60
        inner_height = node_dict["height"]-node_dict["plot_kwargs"]["linewidth"]/60
    else:
        inner_width = node_dict["width"]
        inner_height = node_dict["height"]
    inner = utils.get_node_coordinates(node_dict["center"],inner_radius,inner_width,inner_height)
    # TODO: there seems to be a bug, whenever there is a big linewidth set and the radius is very close to 0,
    #       the inner radius seems to be way bigger than the outer. Don't understand where this comes from.

    outer_kwargs = node_dict["plot_kwargs"].copy()
    outer_kwargs["linewidth"] = 0
    inner_kwargs = node_dict["plot_kwargs"].copy()
    inner_kwargs["linewidth"] = 0
    inner_kwargs["zorder"] = node_dict["plot_kwargs"]["zorder"]+.5
    inner_kwargs["color"] = node_dict["fill_color"]

    if node_dict["plot_kwargs"]["linewidth"] > 0 and node_dict["radius"] > 0:
        ax.fill_between(outer[0,:50],outer[1,:50],outer[1,50:100],**outer_kwargs)
        ax.fill_between(inner[0,:50],inner[1,:50],inner[1,50:100],**inner_kwargs)

    if node_dict["text"] is not None:
        ax.text(*node_dict["center"],node_dict["text"],**node_dict["text_kwargs"])


def draw_edge(node_dict,ax):
    """
    Plots a node or edge and its label text (if provided) on a matplotlib axis.

    Parameters
    ----------
    node_dict : dict
        A dictionary containing information about the node, as generated by `create_node`.

    ax : plt.Axes object
        The matplotlib Axes object to plot the node on.
    """

    # plot edge
    ax.plot(*node_dict["points"],**node_dict["plot_kwargs"],antialiased=True)
    arrow_kwargs = node_dict["plot_kwargs"].copy()
    arrow_kwargs["linestyle"] = "-"
    # plot arrow
    ax.plot(*node_dict["arrow_points"],**arrow_kwargs)
    if node_dict["text"] is not None:
        ax.text(*node_dict["edge_center"]+node_dict["text_offset"],node_dict["text"],**node_dict["text_kwargs"])

def create_edge(N1,N2,ax=None,text=None,curve=None,bezier_calculation=None,
                line=None,offset=None,symmetry=None,arrow=None,arrow_angle=None,
                arrow_length=None,rectangle_angle=None,text_offset=None,
                plot_kwargs={},text_kwargs={}):
    """
    Creates edge dictionary and optionally plots edge.

    Parameters
    ----------
    N1, N2 : dict or tuple
        Node dictionaries or coordinate tuples. If coordinate tuples are provided,
        internally a node dictionary with radius 0 is created.
    ax : matplotlib axis, optional
        If not None, the edge is plotted on this axis.
    text : str, optional
        Text to plot next to the edge.
    curve : float or list of tuple, optional
        Curvature of the edge: +1 is a positive-right angled curvature, -1 is a negative one.
    bezier_calculation : {'rectangle', 'rotation'}, optional
        Method of automatic Bezier point calculation.
        'rectangle' calculates the Bezier point as the diagonal points of an axis-aligned rectangle.
        'rotation' calculates the Bezier point by rotating the node center points around their mean by 90 degrees.
    line : float, optional
        Length of straight line of edge before entering the node.
    offset : float, optional
        Space between intersecting node and edge lines. Usually needed at larger linewidths.
    symmetry : bool, optional
        If True, the curvature of the edge is calculated as if \|x1-x2\| == \|y1-y2\|. Attention: the order of N1 and N2 matters with symmetry! The value of 'line' is next to N1 is preserved, the value of 'line' next to N2 is not.
    arrow : {'-', '->', '<-', '<->'}, optional
        Arrow style for the edge.
    arrow_angle : float, optional
        Angle of the arrow in degrees.
    arrow_length : float, optional
        Length of the arrow.
    rectangle_angle : float, optional
        Angle that the 'bezier_calculation'='rectangle' is calculated in degrees.
    text_offset : tuple of float, optional
        (x, y) coordinates to offset the text from the edge center.
    plot_kwargs : dict, optional
        Keyword arguments for plotting the edge with plt.plot().
    text_kwargs : dict, optional
        Keyword arguments for plotting the text with plt.text().

    Returns
    -------
    edge_dict : dict
        Dictionary containing edge information.
    """

    #--- ABBREVIATIONS ---#
    # to make the variables a little bit shorter at least
    # S ... start
    # E ... end
    # pts ... points
    # pt ... point
    # B ... bezier

    #--- CONVERT COORDINATES INTO NODE ---#
    if type(N1) != dict:
        N1 = create_node(N1,radius=0)
    if type(N2) != dict:
        N2 = create_node(N2,radius=0)

    #--- LOAD DEFAULT PARAMETERS & UPDATE KWARGS ---#
    if offset is None:
        S_offset = default.parameters["edge_offset"] if N1['radius'] > 0 else 0
        E_offset = default.parameters["edge_offset"] if N2['radius'] > 0 else 0
    else:
        S_offset = E_offset = offset
    curve = curve if curve is not None else default.parameters["edge_curve"]
    B_calculation = bezier_calculation if bezier_calculation is not None else default.parameters["edge_bezier_calculation"]
    line = line if line is not None else default.parameters["edge_line"]
    symmetry = symmetry if symmetry is not None else default.parameters["edge_symmetry"]
    arrow = arrow if arrow is not None else default.parameters["edge_arrow"]
    arrow_angle = arrow_angle if arrow_angle is not None else default.parameters["edge_arrow_angle"]
    rectangle_angle = rectangle_angle if rectangle_angle is not None else default.parameters["edge_rectangle_angle"]
    arrow_length = arrow_length if arrow_length is not None else default.parameters["edge_arrow_length"]
    text_offset = text_offset if text_offset is not None else default.parameters["edge_text_offset"]
    plot_kwargs, = utils.separate_list_kwarg_dicts(plot_kwargs,"edgeplot_",[],1)
    text_kwargs, = utils.separate_list_kwarg_dicts(text_kwargs,"edgetext_",[],1)

    #--- CALCULATE POINTS OF THE EDGE ---#
    if type(curve)==list:
        B_pts = np.array(curve)
    elif type(curve)==float or type(curve)==int:
        if B_calculation == "rectangle":
            B_pt = utils.calculate_single_symmetric_bezier_point(N1["center"],N2["center"],curve,rectangle_angle)
        elif B_calculation == "rotation":
            B_pt = utils.calculate_single_rotated_bezier_point(N1["center"],N2["center"],curve)
        else:
            logging.warning(f"'{B_calculation}' is not implemented, defaults to 'rectangle'.")
            B_pt = utils.calculate_single_symmetric_bezier_point(N1["center"],N2["center"],curve,rectangle_angle)
        B_pts = [B_pt]
    else:
        # TODO: Implement curve of numpy array type.
        raise TypeError(f"'curve' argument requires either real or list of x,y-tuples, instead got {type(curve).__name__}.")

    B_pts = np.array(B_pts)
    B_pts_mask = np.ones(len(B_pts),dtype=bool)


    S_pt, _ = utils.calculate_virtual_bezier_point(N1,N2,B_pts[ 0],line=S_offset)
    E_pt, _ = utils.calculate_virtual_bezier_point(N2,N1,B_pts[-1],line=E_offset)
    virtual_S_pt, is_inside_virtual_S  = utils.calculate_virtual_bezier_point(N1,N2,B_pts[ 0],line=line+S_offset)
    if is_inside_virtual_S:
        B_pts_mask[ 0] = False
    if symmetry:
        S_distance = np.linalg.norm(virtual_S_pt-B_pts[0])
        E_vector = E_pt-B_pts[-1]
        norm_E_vector = np.linalg.norm(E_vector)
        virtual_E_pt = B_pts[-1]+E_vector/norm_E_vector*S_distance
        is_inside_virtual_E = False
    else:
        virtual_E_pt,is_inside_virtual_E = utils.calculate_virtual_bezier_point(N2,N1,B_pts[-1],line=line+E_offset)
        if is_inside_virtual_E:
            B_pts_mask[-1] = False
    all_B_pts = np.vstack([virtual_S_pt,*B_pts[B_pts_mask],virtual_E_pt])
    pts = utils.bezier_curve(all_B_pts)
    pts = np.hstack([np.array([S_pt]).T,pts,np.array([E_pt]).T])

    # if debug mode is no, plot bezier points
    if default.parameters["edge_debug"] and ax is not None:
        for p in [S_pt,virtual_S_pt,virtual_E_pt,E_pt]:
            ax.scatter(*p,marker="x",zorder=10)
        for p in B_pts[B_pts_mask]:
            ax.scatter(*p,marker="o",zorder=10)


    #--- CALULATE ARROW ---#
    nan_array = np.array([[np.nan,np.nan]]).T
    arrow_pts = nan_array
    # TODO: There is a bug, when the 2 nodes are too close together, the direction of the arrow flips,
    #       but only in the calculation of the backwards arrow.
    #       It seems like I cannot replicate this bug anymore. Did I fix this by accident?
    for f in [1,-1]:
        # Arrow at start node
        if is_inside_virtual_S:
            # If the bezier point is inside the node, the arrow would point outwards.
            # This is of course not wanted, so we change the arrow_angle by 180°.
            tmp_arrow_angle = arrow_angle-180
        else:
            tmp_arrow_angle = arrow_angle

        R1 = utils.rotate_point(B_pts[0],S_pt,tmp_arrow_angle*f)
        vecR1 = R1-S_pt
        normR1 = np.linalg.norm(vecR1)
        corrR1 = S_pt+vecR1/normR1*arrow_length
        arrow_pts = np.hstack([arrow_pts,np.swapaxes([corrR1,S_pt],1,0),nan_array])
        # Arrow at end node
        if is_inside_virtual_E:
            # If the bezier point is inside the node, the arrow would point outwards.
            # This is of course not wanted, so we change the arrow_angle by 180°.
            tmp_arrow_angle = arrow_angle-180
        else:
            tmp_arrow_angle = arrow_angle
        R1 = utils.rotate_point(B_pts[-1],E_pt,tmp_arrow_angle*f)
        vecR1 = R1-E_pt
        normR1 = np.linalg.norm(vecR1)
        corrR1 = E_pt+vecR1/normR1*arrow_length
        arrow_pts = np.hstack([arrow_pts,np.swapaxes([corrR1,E_pt],1,0),nan_array])

    if arrow == "<-":
        arrow_pts = arrow_pts[:,[1,2,3,7,8]]
    elif arrow == "->":
        arrow_pts = arrow_pts[:,[4,5,6,10,11]]
    elif arrow == "<->":
        arrow_pts = arrow_pts[:,1:-1]
    elif arrow == "-":
        arrow_pts = arrow_pts[:,[]]
    else:
        logging.warning(f"Arrow style {arrow} is not implemented. Defaults to '-'.")
        arrow_pts = arrow_pts[:,[]]

    edge_dict = {
        "points": pts,
        "arrow_points": arrow_pts,
        "plot_kwargs": plot_kwargs,
        "text": text,
        "text_kwargs": text_kwargs,
        "start_node_center": N1["center"],
        "end_node_center": N2["center"],
        "curve": curve,
        "text_offset": text_offset,
    }

    edge_dict["edge_center"] = utils.calculate_edge_center(edge_dict)

    if ax is not None:
        draw_edge(edge_dict,ax)
    return edge_dict

def plot_system_bounds(xlim,ylim,r,ax=None,plot_kwargs={}):
    """
    Plot a systemboundary between xlim and ylim.

    Parameters
    ----------
    xlim : tuple or list
        Min and max value of system bounds along X-axis.
    ylim : tuple or list
        Min and max value of system bounds along y-axis.
    r : float
        Radius of the corner of the bounds.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object to plot the points on. Defaults to None.
    plot_kwargs : dict, optional
        A dictionary of keyword arguments for customizing the system boundary's plot style.
        Defaults to an empty dictionary.

    Returns
    -------
    a : Numpy.NDarray
        Array of shape (2, :) with X and Y coordinates of points of the system boundary.
    """

    a = utils.get_system_bounds(xlim,ylim,r)

    if ax is not None:
        ax.plot(a[0,:],a[1,:],**plot_kwargs)
    return a