# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:37:52 2024

@author: jgh6ds
"""

import kineticstoolkit.lab as ktk
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import os


def interconn(sub, color_tuple):
    interconnections = dict()  # Will contain all segment definitions
    interconnections["{}_LLowerLimb".format(sub)] = {
        "Color": color_tuple,  # In RGB format (here, greenish blue)
        "Links": [  # List of lines that span lists of markers
            ["{}:LTOE".format(sub), "{}:LHEE".format(sub), "{}:LANK".format(sub), "{}:LTOE".format(sub)],
            ["{}:LANK".format(sub), "{}:LKNE".format(sub), "{}:LASI".format(sub)],
            ["{}:LKNE".format(sub), "{}:LPSI".format(sub)],
        ],
    }

    interconnections["{}_RLowerLimb".format(sub)] = {
        "Color": color_tuple,
        "Links": [
            ["{}:RTOE".format(sub), "{}:RHEE".format(sub), "{}:RANK".format(sub), "{}:RTOE".format(sub)],
            ["{}:RANK".format(sub), "{}:RKNE".format(sub), "{}:RASI".format(sub)],
            ["{}:RKNE".format(sub), "{}:RPSI".format(sub)],
        ],
    }

    interconnections["{}_LUpperLimb".format(sub)] = {
        "Color": color_tuple,
        "Links": [
            ["{}:LSHO".format(sub), "{}:LELB".format(sub), "{}:LWRA".format(sub), "{}:LFIN".format(sub)],
            ["{}:LELB".format(sub), "{}:LWRB".format(sub), "{}:LFIN".format(sub)],
            ["{}:LWRA".format(sub), "{}:LWRB".format(sub)],
        ],
    }

    interconnections["{}_RUpperLimb".format(sub)] = {
        "Color": color_tuple,
        "Links": [
            ["{}:RSHO".format(sub), "{}:RELB".format(sub), "{}:RWRA".format(sub), "{}:RFIN".format(sub)],
            ["{}:RELB".format(sub), "{}:RWRB".format(sub), "{}:RFIN".format(sub)],
            ["{}:RWRA".format(sub), "{}:RWRB".format(sub)],
        ],
    }

    interconnections["{}_Head".format(sub)] = {
        "Color": color_tuple,
        "Links": [
            ["{}:C7".format(sub), "{}:LFHD".format(sub), "{}:RFHD".format(sub), "{}:C7".format(sub)],
            ["{}:C7".format(sub), "{}:LBHD".format(sub), "{}:RBHD".format(sub), "{}:C7".format(sub)],
            ["{}:LBHD".format(sub), "{}:LFHD".format(sub)],
            ["{}:RBHD".format(sub), "{}:RFHD".format(sub)],
        ],
    }

    interconnections["{}_TrunkPelvis".format(sub)] = {
        "Color": color_tuple,
        "Links": [
            ["{}:LASI".format(sub), "{}:STRN".format(sub), "{}:RASI".format(sub)],
            ["{}:STRN".format(sub), "{}:CLAV".format(sub)],
            ["{}:LPSI".format(sub), "{}:T10".format(sub), "{}:RPSI".format(sub)],
            ["{}:T10".format(sub), "{}:C7".format(sub)],
            ["{}:LASI".format(sub), "{}:LSHO".format(sub), "{}:LPSI".format(sub)],
            ["{}:RASI".format(sub), "{}:RSHO".format(sub), "{}:RPSI".format(sub)],
            ["{}:LPSI".format(sub), "{}:LASI".format(sub), "{}:RASI".format(sub), "{}:RPSI".format(sub), "{}:LPSI".format(sub)],
            ["{}:LSHO".format(sub), "{}:CLAV".format(sub), "{}:RSHO".format(sub), "{}:C7".format(sub), "{}:LSHO".format(sub)],
        ],
    }

    return interconnections


def create_marker_dict(beta, sub):
    
    marker_list = [
        "{}:LFHD".format(sub),
        "{}:RFHD".format(sub),
        "{}:LBHD".format(sub),
        "{}:RBHD".format(sub),
        "{}:C7".format(sub),
        # "{}:T10".format(sub),
        # "{}:CLAV".format(sub),
        # "{}:STRN".format(sub),
        "{}:LSHO".format(sub),
        "{}:LELB".format(sub),
        "{}:LWRA".format(sub),
        "{}:LWRB".format(sub),
        "{}:LFIN".format(sub),
        "{}:RSHO".format(sub),
        "{}:RELB".format(sub),
        "{}:RWRA".format(sub),
        "{}:RWRB".format(sub),
        "{}:RFIN".format(sub),
        "{}:LASI".format(sub),
        "{}:RASI".format(sub),
        "{}:LTHI".format(sub),
        "{}:LKNE".format(sub),
        "{}:LTIB".format(sub),
        "{}:LANK".format(sub),
        "{}:LHEE".format(sub),
        "{}:LTOE".format(sub),
        "{}:RTHI".format(sub),
        "{}:RKNE".format(sub),
        "{}:RTIB".format(sub),
        "{}:RANK".format(sub),
        "{}:RHEE".format(sub),
        "{}:RTOE".format(sub),
    ]
    
    landmark_array = beta
    confidence = np.ones((landmark_array.shape[0], 1, landmark_array.shape[2]))
    landmark_w_confidence = np.concatenate(
        (landmark_array, confidence), axis=1
    )
    timeshape = landmark_w_confidence.shape[2]

    landmark_dict = {
        marker: landmark_w_confidence[i].T
        for i, marker in enumerate(marker_list)
    }

    return landmark_dict, timeshape
    

def plotting_betas_landmark(betas, colors, zoom=5):
    # Red, Black, Light Blue, Olive, Blue, Purple, Dark Cyan
    # colors = [(1, 0, 0), (0, 0, 0), (0, 0.5, 1), (0.5, 0.5, 0), (0, 0, 1), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    
    landmark_dict = {}
    interconnections = {}
    for i, beta in enumerate(betas):
        sub = "sub{}".format(i+1)
        # color = colors[i % len(colors)]
        color = colors[i]
        ld, timeshape = create_marker_dict(beta, sub)
        landmark_dict.update(ld)
        interconn_dict = interconn(sub, color)
        interconnections.update(interconn_dict)

    ts = ktk.TimeSeries()  # Create an empty TimeSeries
    ts.time = np.arange(0, timeshape / 100, 0.01)
    ts.data = landmark_dict

    p = ktk.Player(
        ts,
        interconnections=interconnections,
        up="z",
        target=(0.0, 0.0, 0.0),
        grid_color=(0.95, 0.95, 0.95),
        azimuth=3.1416 / 4,
        elevation=3.1416 / 6,
        zoom=zoom,
        background_color="w",
        point_size=10,
        playback_speed = 0.005
    )

    markers = p.get_contents()
    
    for key in markers.data:
        sub = key.split(":")[0]
        color_index = int(sub.replace("sub", "")) - 1
        color = colors[color_index % len(colors)]
        markers = markers.add_data_info(key, "Color", color)
    
    p.set_contents(markers)

    return p

# Example usage:
# betas = [beta1, beta2, beta3, beta4, ...]  # List of beta arrays
# p = plotting_betas_landmark(betas)

