# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:00:27 2021

@author: Brian
"""

import networkx as nx
import numpy as np
import random

#Choose a key from a dictionary using the value as weights
#https://stackoverflow.com/questions/6612769/is-there-a-more-elegant-way-for-unpacking-keys-and-values-of-a-dictionary-into-t/6612821#6612821
#https://stackoverflow.com/questions/2921847/what-does-the-star-and-doublestar-operator-mean-in-a-function-call
def dict_choice(prob_dict):
    items, weights = zip(*prob_dict.items())
    return random.choices(list(items), weights = list(weights))[0]

#Function for randomly choosing edges
def choose_edges(amount, edge_set):
    return random.sample(edge_set, amount)

#Function for getting adjacency matrix and contact list from graph
def get_adj(G):
    #Use .A method to convert matrix to array.
    ##Adjacency matrix only contains 1 and 0, int8 is sufficient
    A = nx.to_numpy_matrix(G).A.astype("int8")
    #Contact list
    C = np.array(np.where(A>0), dtype="int32").T
    return A, C

#Combinatorial MMP
class MMP_Unaggregated:
    #Pass it the model that was generated
    def __init__(self, model):
        self._P_dict = model["P_dict"]
        self._start_states = model["start_states"]
        self._N = model["mob_config"]["N"]
        
        #If no T specified at runtime, will use the same as the mobility process
        self._T = model["mob_config"]["T"]
        
        #Generate list of adjacency matrices as well (warning: uses a lot of memory)
        self._KEEP_ADJ = False
    
    #Increment Markov State
    def _increment_state(self, old_state):
        return dict_choice(self._P_dict[old_state])
    
    #Run model
    def run(self, T = None):
        #Overwrite default T if specified
        if T is not None:
            self._T = T
        
        #Initialize Results
        if self._KEEP_ADJ:
            self.ADJ = list()
        self.CON = list()
        self.STATE = list()
        
        #We can actually save time on metrics
        self.links_trial = list()
        self.added_trial = list()
        self.removed_trial = list()
        self.kept_trial = list()
        
        #Get vars
        T = self._T
        N = self._N
        
        #Set up for link adding
        #Build list of all edges
        all_edges = list()
        for n1 in range(N):
            for n2 in range(n1+1, N):
                all_edges.append((n1, n2))
        
        #Track state of edges. Start all available, none occupied
        available_edges = set(all_edges)
        occupied_edges = set()
        
        #Initialize graph
        G = nx.Graph()
        G.add_nodes_from(range(N))
        
        #Initialize in random start_states based on distribution
        state = dict_choice(self._start_states)
        links, (n_add, n_rem) = state
        
        #Start
        self.STATE.append(state)
        A, C = get_adj(G)
        if self._KEEP_ADJ:
            self.ADJ.append(A)
        self.CON.append(C)
        
        #Metrics
        self.links_trial.append(links)
        
        for i in range(T):
            #Pick edges to add (defer adding until edges have been removed)
            to_add = choose_edges(n_add, available_edges)
            to_rem = choose_edges(n_rem, occupied_edges)
            
            #Remove edges
            for n1, n2 in to_rem:
                G.remove_edge(n1, n2)
                
                occupied_edges.discard((n1, n2))
                occupied_edges.discard((n2, n1))
                available_edges.add((n1, n2))
            
            #Add edges
            for n1, n2, in to_add:
                G.add_edge(n1, n2)
                
                available_edges.discard((n1, n2))
                available_edges.discard((n2, n1))
                occupied_edges.add((n1, n2))
            
            #Save Results
            self.STATE.append(state)
            A, C = get_adj(G)
            if self._KEEP_ADJ:
                self.ADJ.append(A)
            self.CON.append(C)
            
            #Metrics
            self.links_trial.append(links + n_add - n_rem)
            self.added_trial.append(n_add)
            self.removed_trial.append(n_rem)
            self.kept_trial.append(links - n_rem)
            
            #Increment State
            state = self._increment_state(state)
            links, (n_add, n_rem) = state

#Reduced MMP
class MMP_Aggregated:
    #Pass it the model that was generated
    def __init__(self, model):
        self._P_dict = model["P_dict"]
        self._action_sets = model["action_sets"]
        self._start_states = model["start_states"]
        self._N = model["mob_config"]["N"]
        
        #If no T specified at runtime, will use the same as the mobility process
        self._T = model["mob_config"]["T"]
        
        #Generate list of adjacency matrices as well (warning: uses a lot of memory)
        self._KEEP_ADJ = False
    
    #Increment Markov State
    def _increment_state(self, old_state):
        return dict_choice(self._P_dict[old_state])
    
    #Run model
    def run(self, T = None):
        #Overwrite default T if specified
        if T is not None:
            self._T = T
        
        #Initialize Results
        if self._KEEP_ADJ:
            self.ADJ = list()
        self.CON = list()
        self.STATE = list()
        
        #We can actually save time on metrics
        self.links_trial = list()
        self.added_trial = list()
        self.removed_trial = list()
        self.kept_trial = list()
        
        #Get vars
        T = self._T
        N = self._N
        action_sets = self._action_sets
        
        #Set up for link adding
        #Build list of all edges
        all_edges = list()
        for n1 in range(N):
            for n2 in range(n1+1, N):
                all_edges.append((n1, n2))
        
        #Track state of edges. Start all available, none occupied
        available_edges = set(all_edges)
        occupied_edges = set()
        
        #Initialize graph
        G = nx.Graph()
        G.add_nodes_from(range(N))
        
        #Initialize in random start_states based on distribution
        state = dict_choice(self._start_states)
        
        #Start
        self.STATE.append(state)
        A, C = get_adj(G)
        if self._KEEP_ADJ:
            self.ADJ.append(A)
        self.CON.append(C)
        
        #Metrics
        self.links_trial.append(state)
        
        for i in range(T):
            next_state = self._increment_state(state)
            
            #Action set uses link numbers, convert state to links
            action_set = action_sets[state, next_state]
            n_add, n_rem = dict_choice(action_set)
            
            #Pick edges to add (defer adding until edges have been removed)
            to_add = choose_edges(n_add, available_edges)
            to_rem = choose_edges(n_rem, occupied_edges)
            
            #Remove edges
            for n1, n2 in to_rem:
                G.remove_edge(n1, n2)
                
                occupied_edges.discard((n1, n2))
                occupied_edges.discard((n2, n1))
                available_edges.add((n1, n2))
            
            #Add edges
            for n1, n2, in to_add:
                G.add_edge(n1, n2)
                
                available_edges.discard((n1, n2))
                available_edges.discard((n2, n1))
                occupied_edges.add((n1, n2))
            
            state = next_state
            
            #Save Results
            self.STATE.append(state)
            A, C = get_adj(G)
            if self._KEEP_ADJ:
                self.ADJ.append(A)
            self.CON.append(C)
            
            #Metrics
            self.links_trial.append(state)
            self.added_trial.append(n_add)
            self.removed_trial.append(n_rem)
            self.kept_trial.append(state - n_add)
