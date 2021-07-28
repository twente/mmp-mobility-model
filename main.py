# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 10:41:25 2021

@author: Brian
"""

import numpy as np
from sim_helpers import MOB_SIM, calc_metrics, load_trial
from sim_helpers import build_UNAGG, build_AGG, MMP_SIM
from plot_helpers import TRACE, LINK_DIST, line_plot

"""
Simulate mobility and MMP
"""

#N=25 simulation config
config = {
    "N": 25,
    "W": 10,
    "v": 1,
    "d": 1.5,
    "T": 1000,
    "theta": 20,
    "phi": 30,
    "precision": 3,
    "sample": False
}

#Simulate mobility process and run metrics
MOB_SIM(config, trials = 1000, path = "pkls/Paper_N25")
metrics_MOB = calc_metrics("pkls/Paper_N25")

#Build reduced (aggregated) MMP, simulate, run metrics
model_AGG = build_AGG("pkls/Paper_N25")
MMP_SIM("pkls/Paper_N25_AGG", trials = 1000)
metrics_AGG = calc_metrics("pkls/Paper_N25_AGG")

#Build combinatorial (unaggregated) MMP, simulate, run metrics
model_UNAGG = build_UNAGG("pkls/Paper_N25")
MMP_SIM("pkls/Paper_N25_UNAGG", trials = 1000)
metrics_UNAGG = calc_metrics("pkls/Paper_N25_UNAGG")

"""
Plot Distributions
"""
#Args for number of links
args_links = {
    "xlim": [0, 50],
    "xmaj": 10,
    "ylim": [0, 0.09],
    "ymaj": 0.01,
}
#Args for added/removed links
args_net = {
    "xlim": [0, 30],
    "xmaj": 5,
    "ylim": [0, 0.12],
    "ymaj": 0.02,
}

LINK_DIST(metrics_MOB["links"], args_links)
LINK_DIST(metrics_MOB["added"], args_net)
LINK_DIST(metrics_MOB["removed"], args_net)

LINK_DIST(metrics_AGG["links"], args_links)
LINK_DIST(metrics_AGG["added"], args_net)
LINK_DIST(metrics_AGG["removed"], args_net)

LINK_DIST(metrics_UNAGG["links"], args_links)
LINK_DIST(metrics_UNAGG["added"], args_net)
LINK_DIST(metrics_UNAGG["removed"], args_net)

"""
Plot links in one realization
"""
#Args for trace
args_trace = {
    "xlim": [0, 1000],
    "xmaj": 200,
    "ylim": [0, 50],
    "ymaj": 10,
}

CON = load_trial(0, path = "pkls/Paper_N25")["CON"]
TRACE(CON, args_trace)

CON = load_trial(0, path = "pkls/Paper_N25_AGG")["CON"]
TRACE(CON, args_trace)

CON = load_trial(0, path = "pkls/Paper_N25_UNAGG")["CON"]
TRACE(CON, args_trace)

"""
Plot link retention probability
"""
args = {
    "figsize": (9.6, 4.8),
    "xlabel": "Step",
    "ylabel": "Retention Probability",
    
    "ylog": True,
    "legend": True,
    "grid": True,
    
    "xlim": [0, 10],
    "xmaj": 1,
    
    "font_labels": 13,
    "font_legend": 13,
    "font_ticks": 13,
}

#List of dictionaries
data = [
    {
        "y": np.array(metrics_MOB["retention"]).mean(axis=0)[0:11],
        "label": "Mobility",
        "fmt": "o--"
    },
    {
        "y": np.array(metrics_UNAGG["retention"]).mean(axis=0)[0:11],
        "label": "Combinatorial MMP",
        "fmt": "o--"
    },
    {
        "y": np.array(metrics_AGG["retention"]).mean(axis=0)[0:11],
        "label": "Reduced MMP",
        "fmt": "o--"
    }
]
line_plot(data, args)
