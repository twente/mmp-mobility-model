# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 10:41:25 2021

@author: Brian
"""

import os, json, time
import numpy as np
import pandas as pd

import modules.pickle_manager as PickleManager
from modules.movement_network import MovementNetwork
from modules.mmp import MMP_Aggregated, MMP_Unaggregated

#Default save path
SAVE_PATH = "pkls"

#Metafile name
META_NAME = "_meta.json"
#Do not add file extension, handled by PickleManager
METRICS_NAME = "_metrics"
#Do not add file extension, handled by PickleManager
MODEL_NAME = "_model"

"""
JSON helpers
"""

#Dump JSON
def save_json(data: dict, file_path: str) -> None:
    _create_path_from_file(file_path)
    with open(file_path, 'w') as f:
        json.dump(data, f)

#Read JSON
def read_json(file_path: str, silent: bool = True) -> dict:
    if not os.path.exists(file_path):
        if not silent:
            print("JSON does not exist.")
        return None
    else:
        with open(file_path) as f:
            return json.load(f)

#Create a path if it doesn't exist already
def _safe_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

#Creates the required folders for a file_path
def _create_path_from_file(file_path):
    path = os.path.dirname(file_path)
    if path != "" and not os.path.exists(path):
        os.makedirs(path)

"""
Trial helpers
"""

#Get number of figures for padding trial name
def _padding(trials: int) -> int:
    return max(5, len(str(trials-1)))

#Name only, file extension is added by PickleManager
def _trial_name(n: int, padding: int) -> str:
    return "Trial_{}".format(str(n).zfill(padding))

#Load data of single trial (not used internally)
def load_trial(n, path):
    padding = _padding(n)
    file_name = _trial_name(n, padding)
    return PickleManager.load_pkl(file_name, path)

"""
Simulations / MMP
"""

#Mobility Simulation
def MOB_SIM(config, trials = 1000, path = SAVE_PATH, overwrite = False):
    if os.path.exists(os.path.join(path, META_NAME)) and not overwrite:
        print("[MOB_SIM] Aborting: meta file already exists.")
        return
    
    print("[MOB_SIM] Running mobility simulation.")
    #Timing
    start_time = time.time()
    
    #Run trials
    padding = _padding(trials)
    for i in range(trials):
        print("\rTrial {}/{}".format(i+1, trials), end="")
        
        movnet = MovementNetwork(config)
        movnet.run()
        
        #Save results
        file_name = _trial_name(i, padding)
        results = {
            "POS": movnet.POS,
            "CON": movnet.CON
        }
        PickleManager.save_pkl(results, file_name, path)
    print("")#Start newline
    
    #Timing
    end_time = time.time()
    
    #Save metadata
    _meta = {
        "type": "MOB",
        "trials": trials,
        "runtime": end_time - start_time,
        "config": movnet._config
    }
    save_json(_meta, os.path.join(path, META_NAME))

def MMP_SIM(path, trials = 1000, T = None, overwrite = False):
    if not os.path.exists(os.path.join(path, MODEL_NAME + ".pkl")):
        print("[MMP_SIM] Aborting: model file does not exist.")
        return
    
    if os.path.exists(os.path.join(path, META_NAME)) and not overwrite:
        print("[MMP_SIM] Aborting: meta file already exists.")
        return
    
    #Load model and set MMP class based on model type
    _model = PickleManager.load_pkl(MODEL_NAME, path)
    if _model["type"] == "UNAGG":
        model_class = MMP_Unaggregated
    elif _model["type"] == "AGG":
        model_class = MMP_Aggregated
    else:
        print("[MMP_SIM] Aborting: unknown model type.")
        return
    
    print("[MMP_SIM] Simulating MMP.")
    #Timing
    start_time = time.time()
    
    #Initialize metric arrays
    links_all = []
    added_all = []
    removed_all = []
    kept_all = []
    
    #Simulate
    padding = _padding(trials)
    for i in range(trials):
        print("\rTrial {}/{}".format(i+1, trials), end="")
        mmp = model_class(_model)
        mmp.run(T)
        
        #Save results
        file_name = _trial_name(i, padding)
        results = {
            "STATE": mmp.STATE,
            "CON": mmp.CON
        }
        PickleManager.save_pkl(results, file_name, path)
        
        #Append metrics
        links_all.append(mmp.links_trial)
        added_all.append(mmp.added_trial)
        removed_all.append(mmp.removed_trial)
        kept_all.append(mmp.kept_trial)
    print("")#Start newline
    
    #Timing
    end_time = time.time()
    
    #Save Metadata
    _meta = {
        "type": _model["type"],
        "trials": trials,
        "runtime": end_time - start_time,
        "config": {
            "N": mmp._N,
            "T": mmp._T
        }
    }
    save_json(_meta, os.path.join(path, META_NAME))
    
    #SAVE METRICS
    _metrics = {
        "links": links_all,
        "added": added_all,
        "removed": removed_all,
        "kept": kept_all,
        "runtime_links": 0#Need this so we don't overwrite metrics later
    }
    PickleManager.save_pkl(_metrics, METRICS_NAME, path)

"""
Metrics
"""

#Calculate link metrics
def calc_metrics(path, selection = [], overwrite = False):
    if not os.path.exists(os.path.join(path, META_NAME)):
        print("[calc_metrics] Aborting: meta file does not exist.")
        return
    
    #Initialize metrics
    _metrics = dict()
    if os.path.exists(os.path.join(path, METRICS_NAME + ".pkl")):
        _metrics = PickleManager.load_pkl(METRICS_NAME, path)
    
    print("[calc_metrics] Calculating metrics.")
    
    #Flag to see if we calculated any new metrics and need to save
    skipped = True
    
    #Link metrics (links, added, removed, kept)
    if not selection or "links" in selection:
        #Note: bool(_metrics.get(runtime_links)) can be False
        #MMP model calculates link metrics during simulation so the runtime is 0
        #Safer to check the keys so we don't accidentally recalculate metrics
        if not "runtime_links" in _metrics.keys() or overwrite:
            skipped = False
            _link_metrics(path, _metrics)
        else:
            print("Link metrics already calculated. Skipping.")
    
    #K-step link retention probability
    if not selection or "retention" in selection:
        if not "runtime_retention" in _metrics.keys() or overwrite:
            skipped = False
            _retention(path, _metrics)
        else:
            print("Retention probability already calculated. Skipping.")
    
    #Unique links observed
    if not selection or "unique" in selection:
        if not "runtime_unique" in _metrics.keys() or overwrite:
            skipped = False
            _unique(path, _metrics)
            pass
        else:
            print("Unique links already calculated. Skipping.")
    
    #Save results
    if not skipped:
        PickleManager.save_pkl(_metrics, METRICS_NAME, path)
    
    return _metrics

#Calculate link metrics from the contact list of a single trial
def _link_metrics(path, _metrics):
    _meta = read_json(os.path.join(path, META_NAME))
    trials = _meta["trials"]
    T = _meta["config"]["T"]
    
    print("Calculating link metrics.")
    #Timing
    start_time = time.time()
    
    #Initialize metric arrays
    links_all = []
    added_all = []
    removed_all = []
    kept_all = []
    
    #Process trials
    padding = _padding(trials)
    for i in range(trials):
        print("\rTrial {}/{}".format(i+1, trials), end="")
        
        links_trial = []
        added_trial = []
        removed_trial = []
        kept_trial = []
        
        #Load contact list and convert to DataFrames
        file_name = _trial_name(i, padding)
        CON = PickleManager.load_pkl(file_name, path)["CON"]
        for t in range(len(CON)):
            CON[t] = pd.DataFrame(CON[t].astype(int))
        
        #Calculate links
        for t in range(T+1):
            #Number of links
            links = int(CON[t].shape[0]/2)
            links_trial.append(links)
            
            #Number of added/removed/kept if we we are not at the last time step
            if t < T:
                union = CON[t].append(CON[t+1]).drop_duplicates()
                
                #Added
                added = int((len(union) - CON[t].shape[0]) / 2)
                added_trial.append(added)
                
                #Removed
                removed = int((len(union) - CON[t+1].shape[0]) / 2)
                removed_trial.append(removed)
                
                #Kept
                kept = links - removed
                kept_trial.append(kept)
        
        links_all.append(links_trial)
        added_all.append(added_trial)
        removed_all.append(removed_trial)
        kept_all.append(kept_trial)
    print("")#Start newline
    
    #Timing
    end_time = time.time()
    
    #Add to dict
    _metrics["links"] = links_all
    _metrics["added"] = added_all
    _metrics["removed"] = removed_all
    _metrics["kept"] = kept_all
    
    #Runtime
    _metrics["runtime_links"] = end_time - start_time

#WIP_Retention
def _retention(path, _metrics, max_k = 20):
    _meta = read_json(os.path.join(path, META_NAME))
    trials = _meta["trials"]
    
    #Check max step doesn't exceed T
    T = _meta["config"]["T"]
    if T < max_k:
        max_k = T
    
    print("Calculating retention metrics.")
    #Timing
    start_time = time.time()
    
    #Initialize metric arrays
    retention_all = []
    
    #Process trials
    padding = _padding(trials)
    for i in range(trials):
        print("\rTrial {}/{}".format(i+1, trials), end="")
        
        #Load contact list and convert to DataFrames
        file_name = _trial_name(i, padding)
        CON = PickleManager.load_pkl(file_name, path)["CON"]
        for t in range(len(CON)):
            CON[t] = pd.DataFrame(CON[t].astype(int))
        
        #Make a shallow copy of the contacts list for processing
        k_calc = CON.copy()
        #Calculate number of links at each time step in the contact list to use for weighting
        w_calc = list()
        for C in k_calc:
            w_calc.append(C.shape[0]/2)
        
        #Initialize rentention results
        #For 0 steps, the retention probability is defined as 1
        retention_trial = [1]
        
        #Calculate retention probabilities for all steps
        for k in range(1, max_k + 1):
            #Get the duplicated (i.e. retained) links between each two time steps
            for t in range(len(k_calc) - 1):
                merged = k_calc[t].append(k_calc[t+1])
                k_calc[t] = merged[merged.duplicated()]
            #Drop last entry
            del k_calc[-1]
            del w_calc[-1]
            
            #Calculate fraction of links that were retained
            probs = list()
            for t in range(len(k_calc)):
                total = CON[t].shape[0]
                if total == 0:
                    #If no links, manually set retention probability as 0, otherwise we have divide by 0
                    #We weight by the number of links so it will have no impact in the end
                    probs.append(0)
                else:
                    #Number of links is doubled in contact lists because we have n1-n2 as well as n2-1
                    #But since we are dividing them, it corrects itself
                    probs.append(k_calc[t].shape[0] / total)
            
            #Take weighted sum of probabilities
            if np.sum(w_calc) > 0:
                retention_trial.append(np.multiply(probs, w_calc).sum() / np.sum(w_calc))
            else:
                #Edge case: no links, define retention probability as 0
                #This only happens if there are 0 links at all of the starting time steps
                retention_trial.append(0)
        retention_all.append(retention_trial)
    print("")#Start newline
    
    #Timing
    end_time = time.time()
    
    #Add to dict
    _metrics["retention"] = retention_all
    
    #Runtime
    _metrics["runtime_retention"] = end_time - start_time

#Unique links observed
def _unique(path, _metrics):
    _meta = read_json(os.path.join(path, META_NAME))
    trials = _meta["trials"]
    
    N = _meta["config"]["N"]
    max_links = int(N*(N-1)/2)
    
    print("Calculating unique links.")
    #Timing
    start_time = time.time()
    
    #Initialize metric arrays
    unique_all = []
    
    #Process trials
    padding = _padding(trials)
    for i in range(trials):
        print("\rTrial {}/{}".format(i+1, trials), end="")
        
        #Load contact list and convert to DataFrames
        file_name = _trial_name(i, padding)
        CON = PickleManager.load_pkl(file_name, path)["CON"]
        for t in range(len(CON)):
            CON[t] = pd.DataFrame(CON[t].astype(int))
        
        #Temp list for calculating unique links observed
        uni_calc = CON[0]
        
        #Calculate unique links observed
        unique_trial = [int(uni_calc.shape[0]/2)]
        for i in range(1, len(CON)):
            if int(uni_calc.shape[0]/2) < max_links:
                uni_calc = uni_calc.append(CON[i]).drop_duplicates()
                unique_trial.append(int(uni_calc.shape[0]/2))
            else:
                unique_trial.append(max_links)
        
        unique_all.append(unique_trial)
    print("")#Start newline
    
    #Timing
    end_time = time.time()
    
    #Add to dict
    _metrics["unique"] = unique_all
    
    #Runtime
    _metrics["runtime_unique"] = end_time - start_time

"""
Build model
"""

#Construct unaggregated MMP model using all trials of the mobility process
def build_AGG(path, out_path = None, overwrite = False):
    if not os.path.exists(os.path.join(path, META_NAME)):
        print("[build_AGG] Aborting: meta file does not exist.")
        return
    
    #Set default out_path
    if out_path is None:
        out_path = path + "_AGG"
    
    #Check outpath
    if os.path.exists(os.path.join(out_path, MODEL_NAME + ".pkl")) and not overwrite:
        print("[build_AGG] Aborting: model already exists.")
        return PickleManager.load_pkl(MODEL_NAME, out_path)
        #return
    
    #calc_metrics will return the link metrics and run them if necessary
    _metrics = calc_metrics(path, selection = ["links"])
    
    #Read mobility metadata
    _meta = read_json(os.path.join(path, META_NAME))
    trials = _meta["trials"]
    N = _meta["config"]["N"]
    T = _meta["config"]["T"]
    
    print("[build_AGG] Building aggregated MMP.")
    #Timing
    start_time = time.time()
    
    #Dict representation of P matrix
    P_dict = dict()
    
    #Initialize action sets
    action_sets = dict()
    
    #Observed starting states
    start_states = dict()
    
    #Loop over metrics to build P and action sets
    for i in range(trials):
        print("\rTrial {}/{}".format(i+1, trials), end="")
        for t in range(T):
            #Read links. Link is state.
            L_t = _metrics["links"][i][t]
            L_t1 = _metrics["links"][i][t+1]
            transition = (L_t, L_t1)
            
            #Read add/rem
            added = _metrics["added"][i][t]
            removed = _metrics["removed"][i][t]
            action = (added, removed)
            
            #Add to action set
            if not transition in action_sets.keys():
                action_sets[transition] = dict()
            if not action in action_sets[transition].keys():
                action_sets[transition][action] = 0
            action_sets[transition][action] = action_sets[transition][action] + 1
            
            if t == 0:
                if start_states.get(L_t) is None:
                    start_states[L_t] = 0
                start_states[L_t] = start_states[L_t] + 1
            
            #Add to P_dict
            #Increment transition
            if P_dict.get(L_t) is None:
                P_dict[L_t] = dict()
            if P_dict[L_t].get(L_t1) is None:
                P_dict[L_t][L_t1] = 0
            P_dict[L_t][L_t1] = P_dict[L_t][L_t1] + 1
            
            #Also add L_t1 to P_dict to be sure that we include all encountered states.
            if P_dict.get(L_t1) is None:
                P_dict[L_t1] = dict()
    print("")#Start newline
    
    #Number of absorbing states
    absorbing = 0
    
    #We detect absorbing states as states which have no out-transitions or only self-transitions
    #However, if a state's non-self transitions are to absorbing states, then it is also absorbing
    #Use a while loop until we don't detect any new absorbing states
    while True:
        #Find absorbing states
        new_absorbing = set()
        for state, transitions in P_dict.items():
            if len(transitions) == 0:
                #No out-transitions
                new_absorbing.add(state)
            elif len(transitions) == 1 and transitions.get(state) is not None:
                #Only one transition and it is going back to the same state
                new_absorbing.add(state)
        
        #Remove absorbing states
        #Note: need to specify default return None otherwise it throws error
        for state in new_absorbing:
            P_dict.pop(state, None)
        
        #Remove transitions to absorbing states
        for state in new_absorbing:
            for _, transitions in P_dict.items():
                transitions.pop(state, None)
        
        #Count number of removed absorbing states
        num_removed = len(new_absorbing)
        if num_removed > 0:
            absorbing = absorbing + num_removed
            #Don't break, need to check states again because we might have more absorbing states
        else:
            #No more absorbing states, done
            break
    
    #Calculate states statistics
    L_max = int(N*(N-1)/2)
    num_theoretical = L_max + 1
    num_actual = len(P_dict)
    
    state_stats = {
        "theoretical": num_theoretical,
        "actual": num_actual,
        "absorbing": absorbing,
        "empty": num_theoretical - num_actual - absorbing
    }
    
    #Timing
    end_time = time.time()
    
    _model = {
        #Model Metadata
        "type": "AGG",
        "build_time": end_time - start_time,
        "real_build_time": end_time - start_time + _metrics["runtime_links"],
        
        #Mobility Metadata
        "mob_trials": _meta["trials"],
        "mob_config": _meta["config"],
        
        #Model Info
        "P_dict": P_dict,
        "action_sets": action_sets,
        "start_states": start_states,
        "state_stats": state_stats
    }
    PickleManager.save_pkl(_model, MODEL_NAME, out_path)
    
    #Debug
    return _model


#Construct unaggregated MMP model using all trials of the mobility process
def build_UNAGG(path, out_path = None, overwrite = False):
    if not os.path.exists(os.path.join(path, META_NAME)):
        print("[build_UNAGG] Aborting: meta file does not exist.")
        return
    
    #Set default out_path
    if out_path is None:
        out_path = path + "_UNAGG"
    
    #Check outpath
    if os.path.exists(os.path.join(out_path, MODEL_NAME + ".pkl")) and not overwrite:
        print("[build_UNAGG] Aborting: model already exists.")
        return PickleManager.load_pkl(MODEL_NAME, out_path)
        #return
    
    #calc_metrics will return the link metrics and run them if necessary
    _metrics = calc_metrics(path, selection = ["links"])
    
    #Read mobility metadata
    _meta = read_json(os.path.join(path, META_NAME))
    trials = _meta["trials"]
    N = _meta["config"]["N"]
    T = _meta["config"]["T"]
    
    print("[build_UNAGG] Building aggregated MMP.")
    #Timing
    start_time = time.time()
    
    #Dict representation of P matrix
    P_dict = dict()
    
    #Observed starting states
    start_states = dict()
    
    #Loop over metrics to build P and action sets
    for i in range(trials):
        print("\rTrial {}/{}".format(i+1, trials), end="")
        for t in range(T-1):
            #Read links
            L_t = _metrics["links"][i][t]
            L_t1 = _metrics["links"][i][t+1]
            
            #Read add/rem
            added = _metrics["added"][i][t]
            removed = _metrics["removed"][i][t]
            action_t = (added, removed)
            
            #Read add/rem
            added = _metrics["added"][i][t+1]
            removed = _metrics["removed"][i][t+1]
            action_t1 = (added, removed)
            
            #State
            S_t = L_t, action_t
            S_t1 = L_t1, action_t1
            
            if t == 0:
                if start_states.get(S_t) is None:
                    start_states[S_t] = 0
                start_states[S_t] = start_states[S_t] + 1
            
            #Add to P_dict
            #Increment transition
            if P_dict.get(S_t) is None:
                P_dict[S_t] = dict()
            if P_dict[S_t].get(S_t1) is None:
                P_dict[S_t][S_t1] = 0
            P_dict[S_t][S_t1] = P_dict[S_t][S_t1] + 1
            
            #Also add S_t1 to P_dict to be sure that we include all encountered states.
            if P_dict.get(S_t1) is None:
                P_dict[S_t1] = dict()
    print("")#Start newline
    
    #Number of absorbing states
    absorbing = 0
    
    #We detect absorbing states as states which have no out-transitions or only self-transitions
    #However, if a state's non-self transitions are to absorbing states, then it is also absorbing
    #Use a while loop until we don't detect any new absorbing states
    while True:
        #Find absorbing states
        new_absorbing = set()
        for state, transitions in P_dict.items():
            if len(transitions) == 0:
                #No out-transitions
                new_absorbing.add(state)
            elif len(transitions) == 1 and transitions.get(state) is not None:
                #Only one transition and it is going back to the same state
                new_absorbing.add(state)
        
        #Remove absorbing states
        #Note: need to specify default return None otherwise it throws error
        for state in new_absorbing:
            P_dict.pop(state, None)
        
        #Remove transitions to absorbing states
        for state in new_absorbing:
            for _, transitions in P_dict.items():
                transitions.pop(state, None)
        
        #Count number of removed absorbing states
        num_removed = len(new_absorbing)
        if num_removed > 0:
            absorbing = absorbing + num_removed
            #Don't break, need to check states again because we might have more absorbing states
        else:
            #No more absorbing states, done
            break
    
    #Calculate states statistics
    L_max = int(N*(N-1)/2)
    num_theoretical = (L_max ** 3)/6 + (L_max ** 2) + (11 * L_max / 6) + 1
    num_theoretical = int(num_theoretical)
    num_actual = len(P_dict)
    
    state_stats = {
        "theoretical": num_theoretical,
        "actual": num_actual,
        "absorbing": absorbing,
        "empty": num_theoretical - num_actual - absorbing
    }
    
    #Timing
    end_time = time.time()
    
    _model = {
        #Model Metadata
        "type": "UNAGG",
        "build_time": end_time - start_time,
        "real_build_time": end_time - start_time + _metrics["runtime_links"],
        
        #Mobility Metadata
        "mob_trials": _meta["trials"],
        "mob_config": _meta["config"],
        
        #Model Info
        "P_dict": P_dict,
        "start_states": start_states,
        "state_stats": state_stats
    }
    PickleManager.save_pkl(_model, MODEL_NAME, out_path)
    
    #Debug
    return _model
