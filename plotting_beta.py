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


def plotting_beta_landmark(beta, zoom = 5, playspeed=0.12, target=(0.0, 0.0, 0.0)):

    marker_list = [
        "LFHD",
        "RFHD",
        "LBHD",
        "RBHD",
        "C7",
        # "T10",
        # "CLAV",
        # "STRN",
        "LSHO",
        "LELB",
        "LWRA",
        "LWRB",
        "LFIN",
        "RSHO",
        "RELB",
        "RWRA",
        "RWRB",
        "RFIN",
        "LASI",
        "RASI",
        "LTHI",
        "LKNE",
        "LTIB",
        "LANK",
        "LHEE",
        "LTOE",
        "RTHI",
        "RKNE",
        "RTIB",
        "RANK",
        "RHEE",
        "RTOE",
    ]

    interconnections = dict()  # Will contain all segment definitions
    interconnections["LLowerLimb"] = {
        "Color": (0, 0.5, 1),  # In RGB format (here, greenish blue)
        "Links": [  # List of lines that span lists of markers
            ["*LTOE", "*LHEE", "*LANK", "*LTOE"],
            ["*LANK", "*LKNE", "*LASI"],
            ["*LKNE", "*LPSI"],
        ],
    }

    interconnections["RLowerLimb"] = {
        "Color": (1, 0.5, 0),
        "Links": [
            ["*RTOE", "*RHEE", "*RANK", "*RTOE"],
            ["*RANK", "*RKNE", "*RASI"],
            ["*RKNE", "*RPSI"],
        ],
    }

    interconnections["LUpperLimb"] = {
        "Color": (0, 0.5, 1),
        "Links": [
            ["*LSHO", "*LELB", "*LWRA", "*LFIN"],
            ["*LELB", "*LWRB", "*LFIN"],
            ["*LWRA", "*LWRB"],
        ],
    }

    interconnections["RUpperLimb"] = {
        "Color": (1, 0.5, 0),
        "Links": [
            ["*RSHO", "*RELB", "*RWRA", "*RFIN"],
            ["*RELB", "*RWRB", "*RFIN"],
            ["*RWRA", "*RWRB"],
        ],
    }

    interconnections["Head"] = {
        "Color": (1, 0.5, 1),
        "Links": [
            ["*C7", "*LFHD", "*RFHD", "*C7"],
            ["*C7", "*LBHD", "*RBHD", "*C7"],
            ["*LBHD", "*LFHD"],
            ["*RBHD", "*RFHD"],
        ],
    }

    interconnections["TrunkPelvis"] = {
        "Color": (0.5, 1, 0.5),
        "Links": [
            ["*LASI", "*STRN", "*RASI"],
            ["*STRN", "*CLAV"],
            ["*LPSI", "*T10", "*RPSI"],
            ["*T10", "*C7"],
            ["*LASI", "*LSHO", "*LPSI"],
            ["*RASI", "*RSHO", "*RPSI"],
            ["*LPSI", "*LASI", "*RASI", "*RPSI", "*LPSI"],
            ["*LSHO", "*CLAV", "*RSHO", "*C7", "*LSHO"],
        ],
    }

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

    ts = ktk.TimeSeries()  # Create an empty TimeSeries
    ts.time = np.arange(0, timeshape / 100, 0.01)
    ts.data = landmark_dict

    p = ktk.Player(
        ts,
        interconnections=interconnections,
        up="z",
        target=target,
        grid_color=(0.95, 0.95, 0.95),
        azimuth=3.1416 / 4,
        elevation=3.1416 / 6,
        zoom=zoom,
        background_color="w",
        point_size=10,
        default_point_color="r",
        playback_speed = playspeed
    )

    return p