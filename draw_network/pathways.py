import numpy as np
import matplotlib.pyplot as plt
import math
from .utils import *
from .core import *
import logging

def add_coreactants(edge,ax=None,side=None,arrow=None,text=None,width=None,
                    height=None,bez_width=None,
                    edge_kwargs={},node_kwargs={},node_text_kwargs={}):
    """
    Function  that  adds  coreactant  nodes  and  eges to an existing edge. Node
    locations  are  calculated by rotation of the edge center point over 1/4 and
    3/4 of the length of the edge for +/- 90°.

    Parameters
    ----------
    edge : edge-dict
        Edge to which the coreactants are added.
    ax : matplotlib.axes.Axes or None
        matplotlib  axis  on  which  the  nodes  and edges are drawn. If None is
        given, nothing is drawn.
    side : str or None
        Either  "left" or "right" or "both". If None, default parameter is used.
        Side on which the coreactants is drawn as seen from the first node to
        the second node parsed to create_edge() of the edge given.
    text : list of str or None
        List of the node labels. List length should be either 2 for side =
        "left" or "right" or 4 for side = "both". Missing list entries are
        filled with empty strings.
    width : float or None
        Absolute value that pushes the coreactant nodes of the same side apart
         from each other. If None, default parameter is used.
    height: float or None
        Absolute value that pushed the coreactant nodes of opposite sides apart
         from each other. If None, default parameter is used.
    bez_width : float or None
        Compared to coreactant node points, Bezier point locations are shifted
        by bez_width. Entering negative numbers makes the edge curve more
        shallow. If None, default parameter is used.
    edge_kwargs, node_kwargs, node_text_kwargs: dict
        Keyword   arguments  parsed  to  create_edge(**),  create_node(**),  and
        create_node(plot_kwargs=**),  respectively.  Single  entries of the dict
        may  contain lists.  Then  the  options  of  the  list are parsed to the
        create_node/edge  function  in  following  order:  start_left, end_left,
        start_right, end_right. List length should be either 2 for side = "left"
        or "right" or 4 for side = "both".

    Returns
    -------
    nodes : list
        List of (either 2 or 4) nodes depending on the side argument.
    edges : list
        List of (either 2 or 4) edges depending on the side argument.
    """

    #--- LOAD DEFAULT PARAMETERS ---#
    if side is None:
        side = default.parameters["coreactant_side"]
    if arrow is None:
        arrow = default.parameters["coreactant_arrow"]
    if width is None:
        width = default.parameters["coreactant_width"]
    if height is None:
        height = default.parameters["coreactant_height"]
    if bez_width is None:
        bez_width = default.parameters["coreactant_bez_width"]

    #--- DEFINE NUMBER OF EDGES ADDED ---#
    if side == "left" or side == "right":
        arg_list_len = 2
    else:
        arg_list_len = 4

    #--- SEPARATE ALL THE DICTS OF LISTS INTO A LIST OF DICTS ---#
    if "plot_kwargs" not in node_kwargs.keys():
        node_kwargs["plot_kwargs"] = {"linewidth":0}
    if "linewidth" not in node_kwargs["plot_kwargs"].keys():
        node_kwargs["plot_kwargs"]["linewdith"] = 0
    if "radius" not in node_kwargs.keys():
        node_kwargs["radius"] = np.min([height,.2])
    edge_kwargs, node_kwargs = separate_dict_of_lists_into_list_of_dicts(arg_list_len,edge_kwargs,node_kwargs,node_text_kwargs,)

    if text is None:
        text = [""]*arg_list_len
    if len(text) < arg_list_len:
        for i in range(arg_list_len-len(text)):
            text.append("")

    # assure that the sign of width and height is correct
    if edge["start_node_center"][0] > edge["end_node_center"][0]+1e-8:
        width = -width
        height = -height
    # TODO: Sometimes it can still happen that you have to enter negative values of width and height.

    # caluclate coreactant node and bezier point locations
    center, angle, n_locs, b_locs = calculate_coreactant_node_coordinates(edge,side,width=width,height=height)
    if bez_width is not None:
        _center, _angle, _n_locs, b_locs = calculate_coreactant_node_coordinates(edge,side,width=width+bez_width,height=height)
    if arg_list_len == 2:
        edge_kwargs = edge_kwargs * 2
        node_kwargs = node_kwargs * 2
        text = text * 2
        n_locs = n_locs * 2
        b_locs = b_locs * 2

    #--- DEFINE THE ARROWSTYLE ---#
    arrow = separate_arrow(arrow)

    nodes = []
    edges = []
    # left start, end
    if side == "both" or side == "left":
        n_left_start = create_node(n_locs[0],ax=ax,text=text[0],**node_kwargs[0])
        n_left_end   = create_node(n_locs[1],ax=ax,text=text[1],**node_kwargs[1])

        for i in range(2):
            if "curve" not in edge_kwargs[i]:
                edge_kwargs[i]["curve"] = [b_locs[i]]
            if "arrow" not in edge_kwargs[i]:
                edge_kwargs[i]["arrow"] = arrow[i]

        e_left_start = create_edge(n_left_start,center,ax=ax,**edge_kwargs[0])
        e_left_end   = create_edge(center,  n_left_end,ax=ax,**edge_kwargs[1])

        nodes += [n_left_start,n_left_end]
        edges += [e_left_start,e_left_end]

    # right start, end
    if side == "both" or side == "right":
        n_right_start = create_node(n_locs[2],ax=ax,text=text[2],**node_kwargs[2])
        n_right_end   = create_node(n_locs[3],ax=ax,text=text[3],**node_kwargs[3])

        for i in range(2):
            if "curve" not in edge_kwargs[i+2]:
                edge_kwargs[i+2]["curve"] = [b_locs[i]]
            if "arrow" not in edge_kwargs[i+2]:
                edge_kwargs[i+2]["arrow"] = arrow[i]
        e_right_start = create_edge(n_right_start,center,ax=ax,**edge_kwargs[2])
        e_right_end   = create_edge(center,n_right_end  ,ax=ax,**edge_kwargs[3])

        nodes += [n_right_start,n_right_end]
        edges += [e_right_start,e_right_end]

    return nodes, edges

def create_circular_pathway(center,radius,n_nodes,rotation=0,clockwise=True,
                            text=None,ax=None,
                            edge_kwargs={},node_kwargs={},node_text_kwargs={}):
    """
    Creates a circular pathway of nodes and edges.

    Parameters
    ----------
    center : tuple
        Coordinates (x, y) of the center of the circle.
    radius : float
        Radius of the circle.
    n_nodes : int
        Number of nodes to be placed on the circle.
    rotation : float, optional
        Location  where  the  first point should be placed. 0 - right, .5 - top,
        1 - left, 1.5 - bottom.
    clockwise : bool, optional
        Whether the nodes are placed in a clockwise direction, by default True.
    text : list of str of, optional
        Text  labels  for  each  node,  by default None. If None, labels will be 
        empty strings.
    ax : matplotlib.axes.Axes, optional
        matplotlib  axis  on  which  the  nodes  and edges are drawn. If None is
        given, nothing is drawn.
    edge_kwargs, node_kwargs, node_text_kwargs: dict
        Keyword   arguments  parsed  to  create_edge(**),  create_node(**),  and
        create_node(text_kwargs=**),  respectively.  Single  entries of the dict
        may  contain lists.  Then  the  options  of  the  list are parsed to the
        create_node/edge  function  sequentially.

    Returns
    -------
    nodes : list
        List of n_nodes nodes.
    edges : list
        List of n_nodes edges.
    """

    #--- SEPARATE ALL THE DICTS OF LISTS INTO A LIST OF DICTS ---#
    edge_kwargs, node_kwargs = separate_dict_of_lists_into_list_of_dicts(n_nodes,edge_kwargs,node_kwargs,node_text_kwargs,)

    if text is None:
        text = [""]*n_nodes
    if len(text) < n_nodes:
        for i in range(n_nodes-len(text)):
            text.append("")
    coordinates = calculate_circle_coordinates(center,radius,n_nodes,rotation,clockwise)
    circle_nodes = []
    for i, (coord,name) in enumerate(zip(coordinates,text)):
        n = create_node(coord,ax=ax,text=name,**node_kwargs[i])
        circle_nodes.append(n)
    circle_edges = []
    for i in range(len(circle_nodes)-1):
        e = create_edge(*circle_nodes[i:i+2],ax=ax,**edge_kwargs[i])
        circle_edges.append(e)
    e = create_edge(circle_nodes[-1],circle_nodes[0],ax=ax,**edge_kwargs[i+1])
    circle_edges.append(e)

    return circle_nodes, circle_edges
