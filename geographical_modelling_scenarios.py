#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 09:29:01 2020

@author: kerry

This .py file reads in a travel matrix (fixed locations as rows, choice 
locations as columns, values for the route in the relevant position - can be 
either duration, such as minutes, or distance, such as km).

A scenario is taking a subset of the choice locations and allocating the 
nearest one to each fixed location.

All possible scenarios are explored, so will choose from 1 to all of the choice 
locations.

Code returns an output file with these parameters to describe the scenario:
median_travel: median travel for total system
max_travel: max travel for total system
95pctl_travel: 95th percentile travel for total system
activity_within_30: total activity within the treshold travel for total system
location_n: for each choice location, included in the scenario: 1 = open    
median_n: for each choice location: median travel
max_n: for each choice location: max travel
95pctl_n: for each choice location: 95th percentile travel
activity_n: for each choice location: total activity
activity_within_threshold_5: for each choice location: total activity within 
                             the treshold travel

"""

# read in the libraries
import pandas as pd
import numpy as np
import itertools

def setup_dataframe(locations):
    """
    locations: ID's of all the possible locations that will be open or closed 
    in each scenario
    
    Create and return a pandas dataframe that has the location IDs as the index
    """
    df = pd.DataFrame()
    
    #create a single column containing the location IDs
    df["location"] = locations

    #set index as the location IDs
    df.set_index(locations, inplace = True)
    #and remove the column
    df = df.drop("location", axis = 1)

    return (df)

# read in data
activity_data = pd.read_csv("data_etc/activity_data.csv")
travel_matrix = pd.read_csv("output/pivoted_results_time_min.csv")

# get the choice locations (the ones that are going to be open or closed in 
# each scenario)
locations = travel_matrix.columns
locations = locations.drop('from_postcode')

#number of chose locations
n_locations = int(locations.shape[0])

#initiate a pandas dataframe that has a row per fixed locations, and will store 
#two columns per scenario: the travel to the nearest choice location, and the 
#choice location
scenario_results = pd.DataFrame()
scenario_results["from_postcode"] = travel_matrix.from_postcode

#initiate a pandas dataframe that has a row per choice location (set as the 
#index)
scenario_definition = setup_dataframe(locations)

#using itertools, go through each combination of choosing 1 to n of the choice
#locations. These are the scenarios.
#for each scenario calculate and store the travel to the nearest choice 
#location, and the choice location
count = 0
for choose in range(1, n_locations + 1):
    for subset in (itertools.combinations(locations, choose)):
        #per fixed location, store the min travel value
        scenario_results[f"min_travel_{count}"] = (
                                    travel_matrix[list(subset)].min(axis = 1))
        #per fixed location, store the nearest choice location
        scenario_results[f"nearest_location_{count}"] = (
                                travel_matrix[list(subset)].idxmin(axis = 1))
        
        #store the scenario definition (which choice locations are open)
        df = pd.DataFrame(subset).set_index(0)
        df[f"scen_{count}"] = 1
        scenario_definition = scenario_definition.join(df)

        #increment the count
        count += 1

#store the total number of scenarios
n_scenarios = count

#add activity data for each fixed location to the dataframe
scenario_results = \
    scenario_results.merge(activity_data, 
                           on = 'from_postcode', 
                           how = 'inner')
    
#set a threshold value, code returns the amount of activity that travels within 
#this threshold
threshold_value = 30

#output values are stored in a numpy array (output_results_np).
#these values are used to define the size of the numpy array/
n_perf_metrics = 4
n_cols = (n_locations * n_perf_metrics) + n_perf_metrics + (2 * n_locations)

#numpy array storing the output values
output_results_np = np.empty((n_scenarios, n_cols))

#initialise lists that will store teh output values
system_median = []
system_max = []
system_95pctl = []
system_activity_threshold = []

#initialise dataframes that will store teh output values
loc_median_df = setup_dataframe(locations)
loc_max_df = setup_dataframe(locations)
loc_95pctl_df = setup_dataframe(locations)
loc_demand_df = setup_dataframe(locations)
loc_demand_threshold_df = setup_dataframe(locations)

#for each scenario, calculate the performance metrics (using the two relevant 
#columns per scenario in the dataframe "scenario_results")
for scen in range(n_scenarios):
    
    #create a new column that stores the activity if it's within the threshold 
    #travel value
    scenario_results[f'activity_within_threshold_{scen}'] = \
            np.where(scenario_results[f'min_travel_{scen}'] < threshold_value, 
                     scenario_results['activity'], 
                     0)

    #calculate performance outputs for total system, store in list
    system_median.append(scenario_results[f"min_travel_{scen}"].median())
    system_max.append(scenario_results[f"min_travel_{scen}"].max())
    system_95pctl.append(scenario_results[f"min_travel_{scen}"].quantile(0.95))
    system_activity_threshold.append(
        scenario_results[f'activity_within_threshold_{scen}'].sum())

    #calculate performance outputs for each fixed location, store in dataframe 
    #using fixed locations as index so as to align the same data
    loc_median_s = (scenario_results.groupby(by = f"nearest_location_{scen}")
                [f"min_travel_{scen}"].median())
    loc_median_df = loc_median_df.join(loc_median_s)

    loc_max_s = (scenario_results.groupby(by = f"nearest_location_{scen}")
             [f"min_travel_{scen}"].max())
    loc_max_df = loc_max_df.join(loc_max_s)

    loc_95pctl_s = (scenario_results.groupby(by = f"nearest_location_{scen}")
             [f"min_travel_{scen}"].quantile(0.95))
    loc_95pctl_df = loc_95pctl_df.join(loc_95pctl_s)

    #demand per dispatch team
    loc_demand_s = (scenario_results.groupby(by = f"nearest_location_{scen}")
                    [f'activity'].sum())
    loc_demand_s = loc_demand_s.rename(f"activity_{scen}") #give unique title
    loc_demand_df = loc_demand_df.join(loc_demand_s)
    
    #demand within threshold
    loc_demand_threshold_s = (scenario_results.groupby
                              (by = f"nearest_location_{scen}")
                              [f'activity_within_threshold_{scen}'].sum())
    loc_demand_threshold_df = (
        loc_demand_threshold_df.join(loc_demand_threshold_s))

#transposing the dataframes, so have scenario per row and performance metrics 
#by column
scenario_definition = scenario_definition.T
loc_median_df = loc_median_df.T
loc_max_df = loc_max_df.T
loc_95pctl_df = loc_95pctl_df.T
loc_demand_df = loc_demand_df.T
loc_demand_threshold_df = loc_demand_threshold_df.T

#create the column titles for the output file
#initialise lists
title0 = []
title1 = []
title2 = []
title3 = []
title4 = []
title5 = []
for loc in range(n_locations):
    title0.append(f"location_{loc}")
    title1.append(f"median_{loc}")
    title2.append(f"max_{loc}")
    title3.append(f"95pctl_{loc}")
    title4.append(f"activity_{loc}")
    title5.append(f"activity_within_threshold_{loc}")

col_titles = ["median_travel", "max_travel", "95pctl_travel", 
              f"activity_within_{threshold_value}"]
for i in title0:
    col_titles.append(i)
for i in title1:
    col_titles.append(i)
for i in title2:
    col_titles.append(i)
for i in title3:
    col_titles.append(i)
for i in title4:
    col_titles.append(i)
for i in title5:
    col_titles.append(i)

#create row labels for the output file
row_titles = []
for i in range(n_scenarios):
    row_titles.append(f"scen_{i}")

#populate the numpy array
output_results_np[:, 0] = system_median
output_results_np[:, 1] = system_max
output_results_np[:, 2] = system_95pctl
output_results_np[:, 3] = system_activity_threshold
output_results_np[:, 4:(4 + n_locations)] = scenario_definition
output_results_np[:, (4 + n_locations):(4 + (2 * n_locations))] = loc_median_df
output_results_np[:, (4 + (2 * n_locations)):(4 + (3 * n_locations))] = \
                                                                    loc_max_df
output_results_np[:, (4 + (3 * n_locations)):(4 + (4 * n_locations))] = \
                                                                loc_95pctl_df
output_results_np[:, (4 + (4 * n_locations)):(4 + (5 * n_locations))] = \
                                                                loc_demand_df
output_results_np[:, (4 + (5 * n_locations)):(4 + (6 * n_locations))] = \
                                                        loc_demand_threshold_df

#convert to dataframe and use the column titles and row labels
output_results_df = pd.DataFrame(data = output_results_np, 
                                 index = row_titles, 
                                 columns = col_titles)

#write output
output_results_df.to_csv("output/gm_scenario_results.csv")