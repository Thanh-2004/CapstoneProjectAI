import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

restaurants = pd.read_csv('foodhygienedata.csv', encoding_errors= 'ignore')

coordinates = [(restaurants['latitude'][i] / (10**4), restaurants['longitude'][i] / (10**4)) for i in range(len(restaurants['latitude'])) if restaurants['longitude'][i] / (10**4) > 32]
coordinates = list(set(coordinates))
copy_coordinates = coordinates.copy()

def lst_key(tup: tuple):
    return tup[1]

def lst_key2(tup: tuple):
    return tup[2]

def haversine(pair: tuple): 
    ''' The distance (km) of any 2 points using their (latitude, longitude) coordinates'''
    x, y = pair
    latx, lonx = x
    laty, lony = y
    
    Earth_radius = 6731
    phi_x = math.radians(latx)
    phi_y = math.radians(laty)
    
    delta_phi = math.radians(laty - latx)
    delta_lambda = math.radians(lony - lonx)
    
    a = math.sin(delta_phi/2)**2 + math.cos(phi_x) * math.cos(phi_y) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    km = Earth_radius * c
    return km

def haversine_transformation(lst: list):
    return list(map(haversine, lst))

def uclidean_distance(pair: tuple):
    x, y = pair
    return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
    
# Numbers of first, second, third and fourth-level points
np.random.seed(2)
solution_space = []
for n1 in range(1, 98):
    for n2 in range(1, 99 - n1):
        for n3 in range(1, 100 - n1 - n2):
            n4 = 100 - (n1 + n2 + n3)
            solution = (n1, n2, n3, n4)
            solution_space.append(solution)
                
n1, n2, n3, n4 = solution_space[np.random.randint(0, len(solution_space), 1)[0]]

# In case, we want to specify the number of points in each level: n1, n2, n3, n4 = 25, 25, 25, 25
# n1, n2, n3, n4 = 25, 25, 25, 25

# Data points
i0 = 0
start, finish = coordinates[1831], (38,33.8)
del coordinates[1831]
while i0 >= 0:
    np.random.seed(i0)
    position1 = tuple(np.random.randint(0, len(coordinates), n1))
    if len(set(position1)) != n1:
        i0 += 1
        continue
    else:
        first_level = [coordinates[i] for i in position1]
        for i in position1:
            del coordinates[i]
                
        position2 = tuple(np.random.randint(0, len(coordinates), n2))
        if len(set(position2)) != n2:
            i0 += 1
            continue
        else:
            second_level = [coordinates[i] for i in position2]
            for i in position2:
                del coordinates[i]
                    
            position3 = tuple(np.random.randint(0, len(coordinates), n3))
            if len(set(position3)) != n3:
                i0 += 1
                continue
            else:
                third_level = [coordinates[i] for i in position3]
                for i in position3:
                    del coordinates[i]
                        
                position4 = tuple(np.random.randint(0, len(coordinates), n4))
                if len(set(position4)) != n4:
                    i0 += 1
                    continue
                else:
                    fourth_level = [coordinates[i] for i in position4]
                    for i in position4:
                        del coordinates[i]
                    break

# These codes are for the conditions for the positions of the petrol
# Distances between each level
pairs0 = [(start, i) for i in first_level]
distances0 = haversine_transformation(pairs0)
min0 = min(distances0)
closest1 = pairs0[distances0.index(min0)][1] # The first level point closest to the STARTING POINT

pairs1 = [] # Pairs (closest1, second level points)
for j in second_level:
    pairs1.append((closest1, j))
distances1 = haversine_transformation(pairs1)
min1 = min(distances1)
closest2 = pairs1[distances1.index(min1)][1] # The second level point closest to closest1

        
pairs2 = [] # Pairs (closest2, third level points)
for j in third_level:
    pairs2.append((closest2, j))
distances2 = haversine_transformation(pairs2)
min2 = min(distances2)

closest3 = pairs2[distances2.index(min2)][1] # The third level point closest to closest2

pairs3 = [] # Pairs (closest3, fourth level points)
for j in fourth_level:
    pairs3.append((closest3, j))
distances3 = haversine_transformation(pairs3)
min3 = min(distances3)
closest4 = pairs3[distances3.index(min3)][1] # The fourth level point closest to closest 3
min4 = haversine((closest4, finish))

min_values = [min0, min1, min2, min3, min4]

# In order to ensure that our problem has solution, we are going to devide the electric stations into 5 groups.
# As the positions of the petrol stations are to ensure that our problem always has solution, each group would only require one satisfied stations.
# The bound for n (the limit distance before it must visit a petrol station) will be found based on the level points and those 5 stations.

# First group
min_sums01 = []
for i in range(len(coordinates)):
    if haversine((coordinates[i], start)) > min0: # Ensuring that S1 < min0
        continue
    else:
        sums01 = []
        for j in range(len(first_level)):
            if haversine((coordinates[i], first_level[j])) <= min0: # Ensuring that S2 < min0
                sums01.append((i, j, haversine((coordinates[i], first_level[j])) + haversine((coordinates[i], start))))
        ordered_sums01 = sorted(sums01, key = lst_key2)
        min_sums01.append(ordered_sums01[0])
    
ordered_min_sums01 = sorted(min_sums01, key = lst_key2)
station1 = coordinates[ordered_min_sums01[0][0]]    
Des1 = first_level[ordered_min_sums01[0][1]]
S1 = haversine((start, station1))
S2 = haversine((station1, Des1))
# Here, constrain01 is [S1, S2]
del coordinates[ordered_min_sums01[0][0]]

# Second group
min_sums02 = []
for i in range(len(coordinates)):
    if haversine((Des1, coordinates[i])) > min1: # Ensuring that S3 <= min1
        continue
    else:
        cmin_values2 = min_values.copy()
        cmin_values2.remove(min2)
        sums02 = []
        for j in range(len(second_level)):
            if haversine((coordinates[i], second_level[j])) <= max(cmin_values2): # Ensuring that S4 < min(i) such that i not 2
                sums02.append((i, j, haversine((Des1, coordinates[i])) + haversine((coordinates[i], second_level[j]))))
        ordered_sums02 = sorted(sums02, key = lst_key2)
        min_sums02.append(ordered_sums02[0])
        
    
ordered_min_sums02 = sorted(min_sums02, key = lst_key2)
station2 = coordinates[ordered_min_sums02[0][0]]
Des2 = second_level[ordered_min_sums02[0][1]]
S3 = haversine((Des1, station2))
S4 = haversine((station2, Des2))
# Here, constraint02 = [S3, S4]
del coordinates[ordered_min_sums02[0][0]]

# Third group
min_sums03 = []
for i in range(len(coordinates)):
    if haversine((Des2, coordinates[i])) > min2: # Ensuring that S5 < min2
        continue
    else:
        cmin_values3 = min_values.copy()
        cmin_values3.remove(min3)
        sums03 = []
        for j in range(len(third_level)):
            if haversine((coordinates[i], third_level[j])) <= max(cmin_values3): # Ensuring that S6 < min(i) such that i not 3
                sums03.append((i, j, haversine((Des2, coordinates[i])) + haversine((coordinates[i], third_level[j]))))
        ordered_sums03 = sorted(sums03, key = lst_key2)
        min_sums03.append(ordered_sums03[0])

ordered_min_sums03 = sorted(min_sums03, key = lst_key2)
station3 = coordinates[ordered_min_sums03[0][0]]
Des3 = third_level[ordered_min_sums03[0][1]]
S5 = haversine((Des2, station3))
S6 = haversine((station3, Des3))
# Here, constraint23 = [S5, S6]
del coordinates[ordered_min_sums03[0][0]]

# # Fourth group:
min_sums04 = []
for i in range(len(coordinates)):
    if haversine((Des3, coordinates[i])) > min3: # Ensuring that S7 <= min3
        continue
    else:
        cmin_values4 = min_values.copy()
        cmin_values4.remove(min4)
        sums04 = []
        for j in range(len(fourth_level)):
            if haversine((coordinates[i], fourth_level[j])) <= sum(cmin_values4): # Ensuring that S8 <= min0 + min1 + min2 + min3
                sums04.append((i, j, haversine((Des3, coordinates[i])) + haversine((coordinates[i], fourth_level[j]))))
        ordered_sums04 = sorted(sums04, key = lst_key2)
        min_sums04.append(ordered_sums04[0])

ordered_min_sums04 = sorted(min_sums04, key = lst_key2)
station4 = coordinates[ordered_min_sums04[0][0]]
Des4 = third_level[ordered_min_sums04[0][1]]
S7 = haversine((Des3, station4))
S8 = haversine((station4, Des4))
# Here, constraint23 = [S7, S8]

del coordinates[ordered_min_sums04[0][0]]
    
# Fifth group
min_sums05 = []
for i in range(len(coordinates)):
    if haversine((Des4, coordinates[i])) > min4: # Ensuring that S9 <= min4
        continue
    else:
        if haversine((finish, coordinates[i])) <= min4: # Ensuring that S10 <= min4
            min_sums05.append((i, haversine((Des4, coordinates[i])) + haversine((finish, coordinates[i]))))
    
ordered_min_sums05 = sorted(min_sums05, key = lst_key)
station5 = coordinates[ordered_min_sums05[0][0]]
S9 = haversine((Des4, station5))
S10 = haversine((station5, finish))

# Here, constraint23 = [S9, S10]
del coordinates[ordered_min_sums05[0][0]]

# Upper bound and lower bounds for n
upper_bound = min0 + min1 + min2 + min3 + min4 
lower_bound = max([S1, S2 + S3, S4 + S5, S6 + S7, S8 + S9, S10])

# Here, n should be between the lower bound (65.58885878622274) and the upper bound (226.07363990088345) 

# Other stations could vary as it wishes
random_station_positions = tuple(np.random.randint(0, len(coordinates), 15))
stations = [coordinates[i] for i in random_station_positions]
stations.extend([station1, station2, station3, station4, station4])

# Generation of a data frame
column = ['Coordinates']
outside = ['Start', 'Finish'] + ['Level_1']*len(first_level) + ['Level_2']*len(second_level) + ['Level_3']*len(third_level) + ['Level_4']*len(fourth_level) + ['Stations']*20

#inside = [0, 1] + [i for i in range(len(first_level))] + [i for i in range(len(second_level))] + [i for i in range(len(third_level))] + [i for i in range(len(fourth_level))] + [i for i in range(20)]

inside = ["Start", "Finish"] + [1 for i in range(len(first_level))] + [2 for i in range(len(second_level))] + [3 for i in range(len(third_level))] + [4 for i in range(len(fourth_level))] + ["S" for i in range(20)]

names = ['Start', 'Finish'] + ['FirstLevel{}'.format(i) for i in range(1, n1 + 1)] + ['SecondLevel{}'.format(i) for i in range(1, n2 + 1)] + ['ThirdLevel{}'.format(i) for i in range(1, n3 + 1)] + ['FourthLevel{}'.format(i) for i in range(1, n4 + 1)] + ['Station{}'.format(i) for i in range(1, 21)]

Data = [start, finish]
for i in [first_level, second_level, third_level, fourth_level, stations]:
    Data.extend(i)

finish_pairs = []
for i in Data:
    finish_pairs.append((i, finish))
distances_to_finish = haversine_transformation(finish_pairs)

new_data = []
for i in range(len(Data)):
    new_data.append((Data[i][0], Data[i][1], distances_to_finish[i]))


hier_index = list(zip(outside, inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)
df = pd.DataFrame(new_data, hier_index, ['Latitude', 'Longitude', 'Heuristic'])
df['Names'] = names
# df.to_csv(r'/Users/nguyentrithanh/Documents/20231/Introduction to AI/CapstoneProject/newdata.csv', index = True)


# # Visualization
# starting_point = df.xs(('Start', ))[['Latitude', 'Longitude']]
# finishing_point = df.xs(('Finish'))[['Latitude', 'Longitude']]
# first_points = df.xs(('First level', ))[['Latitude', 'Longitude']]
# second_points = df.xs(('Second level'))[['Latitude', 'Longitude']]
# third_points = df.xs(('Third level'))[['Latitude', 'Longitude']]
# fourth_points = df.xs('Fourth level')[['Latitude', 'Longitude']]
# stations = df.xs(('Stations', ))[['Latitude', 'Longitude']]

# fig, ax = plt.subplots()
# ax.scatter(starting_point['Latitude'], starting_point['Longitude'], c = 'orange', label = 'Start', edgecolors= 'black', linewidths= 1)
# ax.scatter(finishing_point['Latitude'], finishing_point['Longitude'], c = 'black', label = 'Finish', edgecolors= 'black', linewidths= 1)
# ax.scatter(first_points['Latitude'], first_points['Longitude'], c = 'blue', label = 'First level')
# ax.scatter(second_points['Latitude'], second_points['Longitude'], c = 'red', label = 'Second level')
# ax.scatter(third_points['Latitude'], third_points['Longitude'], c = 'green', label = 'Third level')
# ax.scatter(fourth_points['Latitude'], fourth_points['Longitude'], c = 'purple', label = 'Fourth level')
# ax.scatter(stations['Latitude'], stations['Longitude'], c = 'yellow', label = 'Stations')

# ax.text(start[0], start[1], 'Start', color = 'black')
# ax.text(finish[0], finish[1], 'Finish', color = 'black')

# ax.legend()
# ax.set_title('Map of Possible Paths', fontweight = 'bold')
# ax.set_xlabel('Latitude', fontweight = 'bold', fontstyle = 'italic', labelpad = 5, color = 'blue')
# ax.set_ylabel('Longitude', fontweight = 'bold', fontstyle = 'italic', labelpad = 5, color = 'blue')
# plt.show()