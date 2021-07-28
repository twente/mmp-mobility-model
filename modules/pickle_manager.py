# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 18:37:23 2020

@author: Brian
"""

import pickle
import os
from typing import Any

#Default save path
SAVE_PATH = "pkls"

#Serialize object to binary data
def save_pkl(cucumber: Any, file_name: str, path: str = SAVE_PATH, verbose: bool = False) -> None:
    #Create save path if it doesnt exist already
    if not os.path.exists(path):
        os.makedirs(path)
    
    file_location = os.path.join(path, file_name + ".pkl")
    backup_location = os.path.join(path, file_name + "_PickleTempBackup.pkl")
    
    #If pickle already exists, rename it so we have a backup in case of a pickling error.
    if os.path.exists(file_location):
        if os.path.exists(backup_location):
            os.remove(backup_location)
            print("Old backup removed.")
        os.rename(file_location, backup_location)
    
    #Try to pickle
    try:
        with open(file_location, 'wb') as output:
            pickle.dump(cucumber, output, pickle.HIGHEST_PROTOCOL)
    except pickle.PicklingError as e:
        print("PicklingError: " + str(e))
        print("Object not saved.")
        #Remove the garbage and restore from backup if available
        os.remove(file_location)
        if os.path.exists(backup_location):
            os.rename(backup_location, file_location)
        return
    
    #Remove backup if pickle was successful
    if verbose:
        print("Object saved to {}".format(file_location))
    #Remove backup if it exists
    if os.path.exists(backup_location):
        os.remove(backup_location)

#Load object from file
def load_pkl(file_name: str, path: str = SAVE_PATH, verbose: bool = False) -> Any:
    file_location = os.path.join(path, file_name + ".pkl")
    
    try:
        with open(file_location, 'rb') as input:
            cucumber = pickle.load(input)
    except FileNotFoundError:
        print("File not found: {}".format(file_location))
        print("Object not loaded.")
        return None
    
    if verbose:
        print("Object loaded from {}".format(file_location))
    return cucumber
