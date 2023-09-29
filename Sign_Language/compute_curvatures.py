import pandas as pd
import numpy as np
import sys
sys.path.insert(1, '../')
from pickle import *
import time 
import os.path
import os
import dill as pickle
from tqdm import tqdm
from utils_functions import estimation_GS_group



def compute_list_Y_from_group(df, group):
    list_Y = []
    for gls in group:
        n_rep_Thomas = int(df[(df["glose"]==gls) & (df["name"]=='Thomas')]["n_rep"].values[0])
        for k in range(n_rep_Thomas):
            list_Y.append(df[(df["glose"]==gls) & (df["name"]=='Thomas')]["Rwrist"].values[0][k])
        n_rep_Aliza = int(df[(df["glose"]==gls) & (df["name"]=='Aliza')]["n_rep"].values[0])
        for k in range(n_rep_Aliza):
            list_Y.append(df[(df["glose"]==gls) & (df["name"]=='Aliza')]["Rwrist"].values[0][k])
    return list_Y


# Load data frame of data
df = pd.read_pickle('./Process_Data/LSFtraj_5dis_cutrep.pkl')  

group_1 = ["faire l'amour", 'coiffer', 'Dimanche', 'ne pas aimer', 'Toujours', 'Autrefois', 'avril', 'train', "avoir l'air"] #93
group_2 = ['appeler', 'août', 'avant', 'animal', 'chercher', 'avion', 'besoin', 'asthme', 'bus'] #93
group_3 = ['bouche bée', 'donner', 'avoir', 'bonjour', 'bouche cousue', 'brosse à dent', 'cerise', 'chaud', 'commander(resto)'] #93
group_4 = ['aimer', 'autres', 'bleu', 'boire', 'bonbon', 'chagrin', 'chuchoter', 'cloche'] #93
# group_5 = ['eux trois', 'ils', 'commander(qqun)', 'croire', 'ne pas croire', 'curieux', 'jusqu’à', 'il faut', 'elle'] #93
group_5 = ['eux trois', 'ils', 'commander(qqun)', 'croire', 'ne pas croire', 'curieux', 'jusqu’à', 'il faut'] #92
group_6 = ['après-midi', 'débrouiller', 'éclair', 'enfant', 'désolé', 'décembre', 'jour', 'femme'] #90
group_7 = ['jamais', 'dire', 'essayer', 'faim', 'je', 'janvier', 'jaune', 'Maman'] #93

list_Y_1 = compute_list_Y_from_group(df, group_1)
list_Y_2 = compute_list_Y_from_group(df, group_2)
list_Y_3 = compute_list_Y_from_group(df, group_3)
list_Y_4 = compute_list_Y_from_group(df, group_4)
list_Y_5 = compute_list_Y_from_group(df, group_5)
list_Y_6 = compute_list_Y_from_group(df, group_6)
list_Y_7 = compute_list_Y_from_group(df, group_7)


# trial_03 : un noeud tous les 4 points
# trial_04 : un noeud tous les 3 points


directory = r"results/trial_05/"
filename_base = "results/trial_05/"
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, directory)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)


n_call_bayopt_der = 30
# bounds_lbda = np.array([[1e-09, 1e-05], [1e-09, 1e-05]]) # trial_01
# bounds_lbda = np.array([[1e-12, 1e-06], [1e-12, 1e-06]]) # trial_02
bounds_lbda = np.array([[-12, -6], [-12, -6]]) # trial_05
n_call_bayopt_theta = 30

#### 1

print('Start: Estimation curvatures with GS on group 1.', '\n')

filename = filename_base + "group_1_"
estimation_GS_group(filename, list_Y_1, n_call_bayopt_der, bounds_lbda, n_call_bayopt_theta)

print('End: Estimation curvatures with GS on group 1.', '\n')


# #### 2

# print('Start: Estimation curvatures with GS on group 2.', '\n')

# filename = filename_base + "group_2_"
# estimation_GS_group(filename, list_Y_2, n_call_bayopt_der, bounds_lbda, n_call_bayopt_theta)

# print('End: Estimation curvatures with GS on group 2.', '\n')


# #### 3

# print('Start: Estimation curvatures with GS on group 3.', '\n')

# filename = filename_base + "group_3_"
# estimation_GS_group(filename, list_Y_3, n_call_bayopt_der, bounds_lbda, n_call_bayopt_theta)

# print('End: Estimation curvatures with GS on group 3.', '\n')


# #### 4

# print('Start: Estimation curvatures with GS on group 4.', '\n')

# filename = filename_base + "group_4_"
# estimation_GS_group(filename, list_Y_4, n_call_bayopt_der, bounds_lbda, n_call_bayopt_theta)

# print('End: Estimation curvatures with GS on group 4.', '\n')

# #### 6

# print('Start: Estimation curvatures with GS on group 6.', '\n')

# filename = filename_base + "group_6_"
# estimation_GS_group(filename, list_Y_6, n_call_bayopt_der, bounds_lbda, n_call_bayopt_theta)

# print('End: Estimation curvatures with GS on group 6.', '\n')


# #### 7

# print('Start: Estimation curvatures with GS on group 7.', '\n')

# filename = filename_base + "group_7_"
# estimation_GS_group(filename, list_Y_7, n_call_bayopt_der, bounds_lbda, n_call_bayopt_theta)

# print('End: Estimation curvatures with GS on group 7.', '\n')

# #### 5

# print('Start: Estimation curvatures with GS on group 5.', '\n')

# filename = filename_base + "group_5_"
# estimation_GS_group(filename, list_Y_5, n_call_bayopt_der, bounds_lbda, n_call_bayopt_theta)

# print('End: Estimation curvatures with GS on group 5.', '\n')


