# Hyperparameter Setting

classes_5 = ['1/','2/','3/','4/','5/']
classes_7 = ['1/','2/','3/','4/','5/','6/','7/']
epochs, epochs_CL = 50, 20
batch_size = 256

#default_path_sub_H = "C:/Users/hml76/OneDrive/문서/MATLAB/Data_Hunmin/"
#default_path_sub_H_split1 = "C:/Users/hml76/OneDrive/문서/MATLAB/Data_Hunmin_split1/"
#default_path_sub_H_split2 = "C:/Users/hml76/OneDrive/문서/MATLAB/Data_Hunmin_split2/"

#default_path_sub_X = "C:/Users/hml76/OneDrive/문서/MATLAB/Data_Xianyu/"
#default_path_sub_B = "C:/Users/hml76/OneDrive/문서/MATLAB/Data_Brian/"
#default_path_sub_C = "C:/Users/hml76/OneDrive/문서/MATLAB/Data_Carlson/"
#default_path_sub_H2 = "C:/Users/hml76/OneDrive/문서/MATLAB/Data_Harold/"


default_path_sub_H = "C:/Users/hml76/PycharmProjects/MindForce/data/Data_Hunmin/"
default_path_sub_H_split1 = "C:/Users/hml76/PycharmProjects/MindForce/data/Data_Hunmin_split1/"
default_path_sub_H_split2 = "C:/Users/hml76/PycharmProjects/MindForce/data/Data_Hunmin_split2/"

default_path_sub_X = "C:/Users/hml76/PycharmProjects/MindForce/data/Data_Xianyu/"
default_path_sub_B = "C:/Users/hml76/PycharmProjects/MindForce/data/Data_Brian/"
default_path_sub_C = "C:/Users/hml76/PycharmProjects/MindForce/data/Data_Carlson/"
default_path_sub_H2 = "C:/Users/hml76/PycharmProjects/MindForce/data/Data_Harold/"




Info_sub_H = ['Stand (5/27)', 'Stand (6/18)', 'Stand (6/20)', 'Sit_chair (6/20)', 'Sit_chair_leg_crossed (6/20)', 'Sit_floor (6/20)',
            'Stand (6/20-v2)', 'Sit_chair (6/20-v2)', 'Sit_chair_leg_crossed (6/20-v2)', 'Sit_floor (6/20-v2)',
            'Stand (6/23)', 'Sit_chair (6/23)', 'Sit_chair_leg_crossed (6/23)', 'Sit_floor (6/23)',
            'Stand (6/24)', 'Sit_chair (6/24)', 'Sit_chair_leg_crossed (6/24)', 'Sit_floor (6/24)',
            'Stand (6/26)', 'Sit_chair (6/26)', 'Sit_chair_leg_crossed (6/26)', 'Sit_floor (6/26)',
            'Stand (6/27)', 'Sit_chair (6/27)', 'Sit_chair_leg_crossed (6/27)', 'Sit_floor (6/27)',
            'Stand (6/30)', 'Sit_chair (6/30)', 'Sit_chair_leg_crossed (6/30)', 'Sit_floor (6/30)',
            'Stand (7/1)', 'Sit_chair (7/1)', 'Sit_chair_leg_crossed (7/1)', 'Sit_floor (7/1)',
            'Stand (7/2)',
            'Stand (7/7)', 'Sit_chair (7/7)', 'Sit_chair_leg_crossed (7/7)', 'Sit_floor (7/7)',
            'Stand (7/9)', 'Sit_chair (7/9)', 'Sit_chair_leg_crossed (7/9)', 'Sit_floor (7/9)',
            'Stand (7/10)', 'Sit_chair (7/10)', 'Sit_chair_leg_crossed (7/10)', 'Sit_floor (7/10)',
            'Stand (7/11)', 'Sit_chair (7/11)', 'Sit_chair_leg_crossed (7/11)', 'Sit_floor (7/11)',
            'Stand (7/17)', 'Sit_chair (7/17)', 'Sit_floor (7/17)', 'Stand (7/23)', 'Sit_chair (7/23)', 'Sit_floor (7/23)',
            'Stand (7/24)', 'Sit_chair (7/24)', 'Sit_floor (7/24)', 'Stand (7/25)', 'Sit_chair (7/25)', 'Sit_floor (7/25)',
            'Stand (8/1)', 'Sit_chair (8/1)', 'Sit_floor (8/1)']

# Low quality dataset
#'Sit_chair (7/2)', 'Sit_chair_leg_crossed (7/2)', 'Sit_floor (7/2)', 'Stand (7/3)', 'Sit_chair (7/3)', 'Sit_chair_leg_crossed (7/3)', 'Sit_floor (7/3)'

Info_sub_X = ['Stand (6/24)', 'Sit_chair (6/24)', 'Stand (6/26)', 'Sit_chair (6/26)', 'Stand (6/27)', 'Sit_chair (6/27)', 'Stand (6/30)', 'Sit_chair (6/30)',
              'Stand (7/1)', 'Sit_chair (7/1)', 'Stand (7/2)', 'Sit_chair (7/2)', 'Stand (7/9)', 'Sit_chair (7/9)', 'Stand (7/11)', 'Sit_chair (7/11)',
              'Stand (7/23)', 'Sit_chair (7/23)', 'Stand (8/12)', 'Sit_chair (8/12)', 'Stand (8/13)', 'Sit_chair (8/13)']
Info_sub_B = ['Stand (6/27)', 'Sit_chair (6/27)', 'Stand (7/9)', 'Sit_chair (7/9)', 'Stand (7/10)', 'Sit_chair (7/10)', 'Stand (7/16)', 'Sit_chair (7/16)',
              'Stand (7/17)', 'Sit_chair (7/17)', 'Stand (7/24)', 'Sit_chair (7/24)', 'Stand (8/1)', 'Sit_chair (8/1)']
Info_sub_C = ['Stand (6/30)', 'Sit_chair (6/30)', 'Stand (7/9)', 'Sit_chair (7/9)', 'Stand (7/10)', 'Sit_chair (7/10)', 'Stand (7/11)', 'Sit_chair (7/11)',
              'Stand (7/16)', 'Sit_chair (7/16)', 'Stand (7/17)', 'Sit_chair (7/17)', 'Stand (7/23)', 'Sit_chair (7/23)', 'Stand (7/25)', 'Sit_chair (7/25)',
              'Stand (8/01)', 'Sit_chair (8/01)']
Info_sub_H2 = ['Stand (7/9)', 'Sit_chair (7/9)', 'Stand (7/10)', 'Sit_chair (7/10)', 'Stand (7/11)', 'Sit_chair (7/11)', 'Stand (7/16)', 'Sit_chair (7/16)',
               'Stand (7/17)', 'Sit_chair (7/17)', 'Stand (7/23)', 'Sit_chair (7/23)', 'Stand (7/24)', 'Sit_chair (7/24)', 'Stand (7/25)', 'Sit_chair (7/25)',
               'Stand (8/1)', 'Sit_chair (8/1)']
Info_sub_M = ['Stand (8/1)', 'Sit_chair (8/1)', 'Stand (8/10)', 'Sit_chair (8/10)', 'Stand (8/11)', 'Sit_chair (8/11)', 'Stand (8/12)', 'Sit_chair (8/12)',
              'Stand (8/13)', 'Sit_chair (8/13)',]

#Date / Bluetooth address
dataset_sub_H = ["Exp_2025-05-27/E8331D05289A/", "Exp_2025-06-18/E9AD0E7DCC2B/",
            "Exp_2025-06-20-v1/E9AD0E7DCC2B/", "Exp_2025-06-20-v2/E9AD0E7DCC2B/", "Exp_2025-06-20-v3/E9AD0E7DCC2B/", "Exp_2025-06-20-v4/E9AD0E7DCC2B/",
            "Exp_2025-06-20-v5/E9AD0E7DCC2B/", "Exp_2025-06-20-v6/E9AD0E7DCC2B/", "Exp_2025-06-20-v7/E9AD0E7DCC2B/", "Exp_2025-06-20-v8/E9AD0E7DCC2B/",
            "Exp_2025-06-23-v1/E9AD0E7DCC2B/", "Exp_2025-06-23-v2/E9AD0E7DCC2B/", "Exp_2025-06-23-v3/E9AD0E7DCC2B/", "Exp_2025-06-23-v4/E9AD0E7DCC2B/",
            "Exp_2025-06-24-v1/E9AD0E7DCC2B/", "Exp_2025-06-24-v2/E9AD0E7DCC2B/", "Exp_2025-06-24-v3/E9AD0E7DCC2B/", "Exp_2025-06-24-v4/E9AD0E7DCC2B/",
            "Exp_2025-06-26-v1/E9AD0E7DCC2B/", "Exp_2025-06-26-v2/E9AD0E7DCC2B/", "Exp_2025-06-26-v3/E9AD0E7DCC2B/", "Exp_2025-06-26-v4/E9AD0E7DCC2B/",
            "Exp_2025-06-27-v1/E9AD0E7DCC2B/", "Exp_2025-06-27-v2/E9AD0E7DCC2B/", "Exp_2025-06-27-v3/E9AD0E7DCC2B/", "Exp_2025-06-27-v4/E9AD0E7DCC2B/",
            "Exp_2025-06-30-v1/E9AD0E7DCC2B/", "Exp_2025-06-30-v2/E9AD0E7DCC2B/", "Exp_2025-06-30-v3/E9AD0E7DCC2B/", "Exp_2025-06-30-v4/E9AD0E7DCC2B/",
            "Exp_2025-07-01-v1/E9AD0E7DCC2B/", "Exp_2025-07-01-v2/E9AD0E7DCC2B/", "Exp_2025-07-01-v3/E9AD0E7DCC2B/", "Exp_2025-07-01-v4/E9AD0E7DCC2B/",
            "Exp_2025-07-02-v1/E9AD0E7DCC2B/",
            "Exp_2025-07-07-v1/E9AD0E7DCC2B/", "Exp_2025-07-07-v2/E9AD0E7DCC2B/", "Exp_2025-07-07-v3/E9AD0E7DCC2B/", "Exp_2025-07-07-v4/E9AD0E7DCC2B/",
            "Exp_2025-07-09-v1/E9AD0E7DCC2B/", "Exp_2025-07-09-v2/E9AD0E7DCC2B/", "Exp_2025-07-09-v3/E9AD0E7DCC2B/", "Exp_2025-07-09-v4/E9AD0E7DCC2B/",
            "Exp_2025-07-10-v1/E9AD0E7DCC2B/", "Exp_2025-07-10-v2/E9AD0E7DCC2B/", "Exp_2025-07-10-v3/E9AD0E7DCC2B/", "Exp_2025-07-10-v4/E9AD0E7DCC2B/",
            "Exp_2025-07-11-v1/E9AD0E7DCC2B/", "Exp_2025-07-11-v2/E9AD0E7DCC2B/", "Exp_2025-07-11-v3/E9AD0E7DCC2B/", "Exp_2025-07-11-v4/E9AD0E7DCC2B/",
            "Exp_2025-07-17-v1/E9AD0E7DCC2B/", "Exp_2025-07-17-v2/E9AD0E7DCC2B/", "Exp_2025-07-17-v3/E9AD0E7DCC2B/",
            "Exp_2025-07-23-v1/E9AD0E7DCC2B/", "Exp_2025-07-23-v2/E9AD0E7DCC2B/", "Exp_2025-07-23-v3/E9AD0E7DCC2B/",
            "Exp_2025-07-24-v1/E9AD0E7DCC2B/", "Exp_2025-07-24-v2/E9AD0E7DCC2B/", "Exp_2025-07-24-v3/E9AD0E7DCC2B/",
            "Exp_2025-07-25-v1/E9AD0E7DCC2B/", "Exp_2025-07-25-v2/E9AD0E7DCC2B/", "Exp_2025-07-25-v3/E9AD0E7DCC2B/",
            "Exp_2025-08-01-v1/E9AD0E7DCC2B/", "Exp_2025-08-01-v2/E9AD0E7DCC2B/", "Exp_2025-08-01-v3/E9AD0E7DCC2B/"]
# Low quality
# "Exp_2025-07-02-v2/E9AD0E7DCC2B/", "Exp_2025-07-02-v3/E9AD0E7DCC2B/", "Exp_2025-07-02-v4/E9AD0E7DCC2B/", "Exp_2025-07-03-v1/E9AD0E7DCC2B/", "Exp_2025-07-03-v2/E9AD0E7DCC2B/", "Exp_2025-07-03-v3/E9AD0E7DCC2B/", "Exp_2025-07-03-v4/E9AD0E7DCC2B/",

dataset_sub_X = ["Exp_2025-06-24-v1/E9AD0E7DCC2B/", "Exp_2025-06-24-v2/E9AD0E7DCC2B/", "Exp_2025-06-26-v1/E9AD0E7DCC2B/", "Exp_2025-06-26-v2/E9AD0E7DCC2B/",
                 "Exp_2025-06-27-v1/E9AD0E7DCC2B/", "Exp_2025-06-27-v2/E9AD0E7DCC2B/", "Exp_2025-06-30-v1/FEFFF6FFF5FF/", "Exp_2025-06-30-v2/FEFFF6FFF5FF/",
                 "Exp_2025-07-01-v1/E9AD0E7DCC2B/", "Exp_2025-07-01-v2/E9AD0E7DCC2B/", "Exp_2025-07-02-v1/E9AD0E7DCC2B/", "Exp_2025-07-02-v2/E9AD0E7DCC2B/",
                 "Exp_2025-07-09-v1/E9AD0E7DCC2B/", "Exp_2025-07-09-v2/E9AD0E7DCC2B/", "Exp_2025-07-11-v1/E9AD0E7DCC2B/", "Exp_2025-07-11-v2/E9AD0E7DCC2B/",
                 "Exp_2025-07-23-v1/E9AD0E7DCC2B/", "Exp_2025-07-23-v2/E9AD0E7DCC2B/", "Exp_2025-08-12-v1/E9AD0E7DCC2B/", "Exp_2025-08-12-v2/E9AD0E7DCC2B/",
                 "Exp_2025-08-13-v1/E9AD0E7DCC2B/", "Exp_2025-08-13-v2/E9AD0E7DCC2B/"]

dataset_sub_B = ["Exp_2025-06-27-v1/E9AD0E7DCC2B/", "Exp_2025-06-27-v2/E9AD0E7DCC2B/", "Exp_2025-07-09-v1/E9AD0E7DCC2B/", "Exp_2025-07-09-v2/E9AD0E7DCC2B/",
                 "Exp_2025-07-10-v1/E9AD0E7DCC2B/", "Exp_2025-07-10-v2/E9AD0E7DCC2B/", "Exp_2025-07-16-v1/E9AD0E7DCC2B/", "Exp_2025-07-16-v2/E9AD0E7DCC2B/",
                 "Exp_2025-07-17-v1/E9AD0E7DCC2B/", "Exp_2025-07-17-v2/E9AD0E7DCC2B/", "Exp_2025-07-24-v1/E9AD0E7DCC2B/", "Exp_2025-07-24-v2/E9AD0E7DCC2B/",
                 "Exp_2025-08-01-v1/E9AD0E7DCC2B/", "Exp_2025-08-01-v2/E9AD0E7DCC2B/"]

dataset_sub_C = ["Exp_2025-06-30-v1/E9AD0E7DCC2B/", "Exp_2025-06-30-v2/E9AD0E7DCC2B/", "Exp_2025-07-09-v1/E9AD0E7DCC2B/", "Exp_2025-07-09-v2/E9AD0E7DCC2B/",
                 "Exp_2025-07-10-v1/E9AD0E7DCC2B/", "Exp_2025-07-10-v2/E9AD0E7DCC2B/", "Exp_2025-07-11-v1/E9AD0E7DCC2B/", "Exp_2025-07-11-v2/E9AD0E7DCC2B/",
                 "Exp_2025-07-16-v1/E9AD0E7DCC2B/", "Exp_2025-07-16-v2/E9AD0E7DCC2B/", "Exp_2025-07-17-v1/E9AD0E7DCC2B/", "Exp_2025-07-17-v2/E9AD0E7DCC2B/",
                 "Exp_2025-07-23-v1/E9AD0E7DCC2B/", "Exp_2025-07-23-v2/E9AD0E7DCC2B/", "Exp_2025-07-25-v1/E9AD0E7DCC2B/", "Exp_2025-07-25-v2/E9AD0E7DCC2B/",
                 "Exp_2025-08-01-v1/E9AD0E7DCC2B/", "Exp_2025-08-01-v2/E9AD0E7DCC2B/"]

dataset_sub_H2 = ["Exp_2025-07-09-v1/E9AD0E7DCC2B/", "Exp_2025-07-09-v2/E9AD0E7DCC2B/", "Exp_2025-07-10-v1/E9AD0E7DCC2B/", "Exp_2025-07-10-v2/E9AD0E7DCC2B/",
                  "Exp_2025-07-11-v1/E9AD0E7DCC2B/", "Exp_2025-07-11-v2/E9AD0E7DCC2B/", "Exp_2025-07-16-v1/E9AD0E7DCC2B/", "Exp_2025-07-16-v2/E9AD0E7DCC2B/",
                  "Exp_2025-07-17-v1/E9AD0E7DCC2B/", "Exp_2025-07-17-v2/E9AD0E7DCC2B/", "Exp_2025-07-23-v1/E9AD0E7DCC2B/", "Exp_2025-07-23-v2/E9AD0E7DCC2B/",
                  "Exp_2025-07-24-v1/E9AD0E7DCC2B/", "Exp_2025-07-24-v2/E9AD0E7DCC2B/", "Exp_2025-07-25-v1/E9AD0E7DCC2B/", "Exp_2025-07-25-v2/E9AD0E7DCC2B/",
                  "Exp_2025-08-01-v1/E9AD0E7DCC2B/", "Exp_2025-08-01-v2/E9AD0E7DCC2B/"]