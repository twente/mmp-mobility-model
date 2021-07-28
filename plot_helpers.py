# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 12:35:37 2021

@author: Brian
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import os

#Convert contact list to numbers of links
def C2L(CON):
    y = list()
    for C in CON:
        y.append(int(C.shape[0])/2)
    return y

#Generates trace from contact lists
def TRACE(CON, args = {}):
    y = C2L(CON)
    data = [
        {
            "y": y,
            "fmt": "o",
        }
    ]
    
    default_args = {
        "figsize": (9.6, 4.8),
        "xlabel": "Time Step",
        "ylabel": "Number of Links",
        
        "grid": True,
        
        "xlim": [0, len(CON) - 1],
        
        "font_labels": 13,
        "font_ticks": 13,
    }
    default_args.update(args)
    
    return line_plot(data, args=default_args)

#Link dist helper
def LINK_DIST(data, args = {}):
    binwidth = 1
    default_args = {
        "xlabel": "Number of Links",
        "ylabel": "Relative Frequency",
        
        "font_labels": 15,
        "font_ticks": 15,
        
        #Add another extra binwidth because upper bound of last bin is inclusive
        #https://stackoverflow.com/questions/6986986/bin-size-in-matplotlib-histogram
        "bins": range(np.min(data), np.max(data) + 2*binwidth, binwidth),
    }
    default_args.update(args)
    
    return histogram(data, args=default_args)

#Save figure
def savefig(name, fformats = ["png", "eps", "svg"], path = "plots"):
    if not os.path.exists(path):
        os.makedirs(path)
    
    for fformat in fformats:
        file_name = name + "." + fformat
        plt.savefig(os.path.join(path, file_name) , format=fformat)

"""
General
"""

#Histogram
#https://stackoverflow.com/questions/6986986/bin-size-in-matplotlib-histogram
def histogram(data, args = {}):
    size = args.get("figsize", (6.4, 4.8))
    fig, ax = plt.subplots(figsize=size)
    ax.tick_params(direction="in", which="both")
    
    #Defaults to 10 bins
    items = np.array(data).flatten()
    bins = args.get("bins", 10)
    density = args.get("density", True)
    
    ax.hist(items, bins=bins, density=density)
    
    #Handle arguments stuff
    _plt_arg_handler(ax, args)
    
    #Dupe ticks
    y2 = _dupe_y(ax)
    x2 = _dupe_x(ax)
    
    #Return so we can manipulate them afterwards
    return fig, ax, y2, x2

#Make a line plot
def line_plot(data, args = {}):
    size = args.get("figsize", (6.4, 4.8))
    fig, ax = plt.subplots(figsize=size)
    ax.tick_params(direction="in", which="both")
    
    #Plot
    for series in data:
        x = series.get("x")
        y = series.get("y")
        if x is None:
            x = list(range(len(y)))
        
        fmt = series.get("fmt")
        label = series.get("label")
        color = series.get("color")
        
        if fmt:
            ax.plot(x, y, fmt, label=label, color=color)
        else:
            ax.plot(x, y, label=label, color=color)
    
    #Handle arguments stuff
    _plt_arg_handler(ax, args)
    
    #Dupe ticks
    y2 = _dupe_y(ax)
    x2 = _dupe_x(ax)
    
    #Return so we can manipulate them afterwards
    return fig, ax, y2, x2

#Handle all the plot arguments
def _plt_arg_handler(ax, args):
    #Set title
    if args.get("title"):
        if args.get("font_title"):
            ax.set_title(args.get("title"), fontsize=args.get("font_title"))
        else:
            ax.set_title(args.get("title"))
    
    #Set axis labels
    if args.get("xlabel"):
        if args.get("font_labels"):
            ax.set_xlabel(args.get("xlabel"), fontsize=args.get("font_labels"))
        else:
            ax.set_xlabel(args.get("xlabel"))
    if args.get("ylabel"):
        if args.get("font_labels"):
            ax.set_ylabel(args.get("ylabel"), fontsize=args.get("font_labels"))
        else:
            ax.set_ylabel(args.get("ylabel"))
    
    #Legend
    if args.get("legend"):
        if args.get("font_legend"):
            ax.legend(fontsize=args.get("font_legend"))
        else:
            ax.legend()
    
    #Logscale
    if args.get("xlog"):
        ax.set_xscale("log")
    if args.get("ylog"):
        ax.set_yscale("log")
    
    #Grid
    if args.get("grid"):
        ax.grid(which='major')
    
    #Axis limits
    if args.get("xlim"):
        ax.set_xlim(args.get("xlim"))
    if args.get("ylim"):
        ax.set_ylim(args.get("ylim"))
    
    #Axis major ticks
    if args.get("xmaj"):
        ax.xaxis.set_major_locator(MultipleLocator(args.get("xmaj")))
    if args.get("ymaj"):
        ax.yaxis.set_major_locator(MultipleLocator(args.get("ymaj")))
    
    #Tick font size
    #https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot/14971193#14971193
    if args.get("font_ticks"):
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(args.get("font_ticks"))

#Dupe y-axis
def _dupe_y(ax):
    y2 = ax.twinx()
    y2.set_ylim(ax.get_ylim())#Same limits
    y2.set_yscale(ax.get_yscale())#Same scale
    y2.set_yticks(ax.get_yticks())#Same ticks
    y2.set_yticklabels([])#No labels
    y2.set_ybound(ax.get_ybound())#Copy bounds because plt cuts off a bit
    y2.tick_params(direction="in", which="both")#Ticks inside
    return y2

#Dupe x-axis
def _dupe_x(ax):
    x2 = ax.twiny()
    x2.set_xlim(ax.get_xlim())#Same limits
    x2.set_xscale(ax.get_xscale())#Same scale
    x2.set_xticks(ax.get_xticks())#Same ticks
    x2.set_xticklabels([])#No labels
    x2.set_xbound(ax.get_xbound())#Copy bounds because plt cuts off a bit
    x2.tick_params(direction="in", which="both")#Ticks inside
    return x2
