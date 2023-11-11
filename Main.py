import numpy as np
import pandas as pd
import queue
import Data

AllData = Data.df


interval1 = (2, AllData.loc["Level_1"].shape[0] + 2)
interval2 = (interval1[1], interval1[1] + AllData.loc["Level_2"].shape[0])
interval3 = (interval2[1], interval2[1] + AllData.loc["Level_3"].shape[0])
interval4 = (interval3[1], interval3[1] + AllData.loc["Level_4"].shape[0])
intervalS = (interval4[1], interval4[1] + AllData.loc["Stations"].shape[0])

Intervals = [interval1, interval2, interval3, interval4, (0,1)]

def interval_norm(interval):
    norm  = interval[1] - interval[0]
    return norm


def MakeGraphStart_1(df):
    graph = {}
    lst = []
    for i in range (interval1[0], interval1[1]):
        lst.append(Data.haversine(((df["Latitude"].iloc[0], df["Longitude"].iloc[0]) ,
                                   (df["Latitude"].iloc[i], df["Longitude"].iloc[i]))))
    graph[f"Start"] = lst
    return graph
graphS1 = MakeGraphStart_1(AllData)
dfS1 = pd.DataFrame(graphS1, index = [f"FirstLevel{i}" for i in range (1, interval_norm(interval1) +1)])

def MakeGraph1_2(df):
    graph = {}
    lst = []
    for i in range (interval1[0], interval1[1]):
        for j in range (interval2[0], interval2[1]):
            lst.append(Data.haversine(((df["Latitude"].iloc[i],df["Longitude"].iloc[i]), 
                                       (df["Latitude"].iloc[j],df["Longitude"].iloc[j]))))
        graph[f"FirstLevel{i-1}"] = lst
        lst = []

    return graph
graph12 = MakeGraph1_2(AllData)
df12 = pd.DataFrame(graph12, [f"SecondLevel{i}" for i in range (1 , interval_norm(interval2) + 1)])


def MakeGraph2_3(df):
    graph = {}
    lst = []
    for i in range (interval2[0], interval2[1]):
        for j in range (interval3[0], interval3[1]):
            lst.append(Data.haversine(((df["Latitude"].iloc[i],df["Longitude"].iloc[i]), 
                                       (df["Latitude"].iloc[j],df["Longitude"].iloc[j]))))
        graph[f"SecondLevel{i-interval_norm(interval1) - 1}"] = lst
        lst = []

    return graph
graph23 = MakeGraph2_3(AllData)
df23 = pd.DataFrame(graph23, [f"ThirdLevel{i}" for i in range (1 , interval_norm(interval3) + 1)])



def MakeGraph3_4(df):
    graph = {}
    lst = []
    for i in range (interval3[0], interval3[1]):
        for j in range (interval4[0], interval4[1]):
            lst.append(Data.haversine(((df["Latitude"].iloc[i],df["Longitude"].iloc[i]), 
                                       (df["Latitude"].iloc[j],df["Longitude"].iloc[j]))))
        graph[f"ThirdLevel{i-interval_norm(interval1) - interval_norm(interval2) - 1}"] = lst
        lst = []

    return graph
graph34 = MakeGraph3_4(AllData)
df34 = pd.DataFrame(graph34, [f"FourthLevel{i}" for i in range (1 , interval_norm(interval4) + 1)])


def MakeGraph4_Finish(df):
    graph = {}
    lst = []
    for i in range (interval4[0], interval4[1]):
        lst.append(Data.haversine(((df["Latitude"].iloc[i],df["Longitude"].iloc[i]), 
                                   (df["Latitude"].iloc[1],df["Longitude"].iloc[1]))))
        graph[f"FourthLevel{i-interval_norm(interval1) - interval_norm(interval2)- interval_norm(interval3) - 1}"] = lst
        lst = []

    return graph
graph4F = MakeGraph4_Finish(AllData)
df4F = pd.DataFrame(graph4F, ["Finish"])


# station = (AllData["Latitude"].loc["Stations"][0], AllData["Longitude"].loc["Stations"][0])


def MakeGraph1_Station(df):
    graph = {}
    lst = []
    for i in range (interval1[0], interval1[1]):
        for j in range (intervalS[0], intervalS[1]):
            lst.append(Data.haversine(((df["Latitude"].iloc[i],df["Longitude"].iloc[i]), 
                                       (df["Latitude"].iloc[j],df["Longitude"].iloc[j]))))
        graph[f"FirstLevel{i - 1}"] = lst
        lst = []

    return graph

def MakeGraph2_Station(df):
    graph = {}
    lst = []
    for i in range (interval2[0], interval2[1]):
        for j in range (intervalS[0], intervalS[1]):
            lst.append(Data.haversine(((df["Latitude"].iloc[i],df["Longitude"].iloc[i]), 
                                       (df["Latitude"].iloc[j],df["Longitude"].iloc[j]))))
        graph[f"SecondLevel{i -interval_norm(interval1) - 1}"] = lst
        lst = []

    return graph

def MakeGraph3_Station(df):
    graph = {}
    lst = []
    for i in range (interval3[0], interval3[1]):
        for j in range (intervalS[0], intervalS[1]):
            lst.append(Data.haversine(((df["Latitude"].iloc[i],df["Longitude"].iloc[i]), 
                                       (df["Latitude"].iloc[j],df["Longitude"].iloc[j]))))
        graph[f"ThirdLevel{i -interval_norm(interval1) - interval_norm(interval2) - 1}"] = lst
        lst = []

    return graph

def MakeGraph4_Station(df):
    graph = {}
    lst = []
    for i in range (interval4[0], interval4[1]):
        for j in range (intervalS[0], intervalS[1]):
            lst.append(Data.haversine(((df["Latitude"].iloc[i],df["Longitude"].iloc[i]), 
                                       (df["Latitude"].iloc[j],df["Longitude"].iloc[j]))))
        graph[f"FourthLevel{i -interval_norm(interval1) - interval_norm(interval2)- interval_norm(interval3) - 1}"] = lst
        lst = []

    return graph

def MakeGraphStation_Finish(df):
    graph = {}
    lst = []
    for j in range (intervalS[0], intervalS[1]):
        lst.append(Data.haversine(((df["Latitude"].iloc[1],df["Longitude"].iloc[1]), 
                                   (df["Latitude"].iloc[j],df["Longitude"].iloc[j]))))
    graph["Finish"] = lst
    lst = []

    return graph

def MakeGraphStation_Start(df):
    graph = {}
    lst = []
    for j in range (intervalS[0], intervalS[1]):
        lst.append(Data.haversine(((df["Latitude"].iloc[0],df["Longitude"].iloc[0]), 
                                   (df["Latitude"].iloc[j],df["Longitude"].iloc[j]))))
    graph["Start"] = lst
    lst = []

    return graph


graph1St = MakeGraph1_Station(AllData)
graph2St = MakeGraph2_Station(AllData)
graph3St = MakeGraph3_Station(AllData) 
graph4St = MakeGraph4_Station(AllData) 
graphStF = MakeGraphStation_Finish(AllData)
graphStS = MakeGraphStation_Start(AllData)

dfSt = pd.DataFrame(graph1St, [f"Station{i}" for  i in range (1, interval_norm(intervalS)+ 1)])
dfSt = pd.concat([dfSt, pd.DataFrame(graph2St, [f"Station{i}" for  i in range (1, interval_norm(intervalS)+ 1)])], axis=1)
dfSt = pd.concat([dfSt, pd.DataFrame(graph3St, [f"Station{i}" for  i in range (1, interval_norm(intervalS)+ 1)])], axis=1)
dfSt = pd.concat([dfSt, pd.DataFrame(graph4St, [f"Station{i}" for  i in range (1, interval_norm(intervalS)+ 1)])], axis=1)
dfSt = pd.concat([dfSt, pd.DataFrame(graphStF, [f"Station{i}" for  i in range (1, interval_norm(intervalS)+ 1)])], axis=1)
dfSt = pd.concat([dfSt, pd.DataFrame(graphStS, [f"Station{i}" for  i in range (1, interval_norm(intervalS)+ 1)])], axis=1)
dfSt = dfSt.T
# dfSt["Categories"] = ["Level_1"]* interval_norm(interval1) + \
#                      ["Level_2"]* interval_norm(interval2) + \
#                      ["Level_3"]* interval_norm(interval3) + \
#                      ["Level_4"]* interval_norm(interval4) 


DataFrame = [dfS1, df12, df23, df34, df4F]


def DropStationDataFrame(df, interval):
    for _ in range(interval_norm(interval)):
        df = df.drop(df.index[0], axis = 0)
    return df


def PointPath(node, path, interval, df, AllData, fuel, distance):
    global dfSt
    priorityQueue = queue.PriorityQueue()
    priorityQueue_Station = queue.PriorityQueue()
    for i in range (interval_norm(interval)):
        priorityQueue.put((AllData["Heuristic"].iloc[i+interval[0]] + df[node].iloc[i] + distance,
                          [df[node].index[i], df[node].iloc[i]]))
    while priorityQueue.empty() == False:
        current = priorityQueue.get()[1]
        if current[1] > fuel:
            for i in range (1, interval_norm(intervalS) + 1):
                if dfSt[f"Station{i}"].loc[current[0]] > fuel:
                    continue
                else:
                    priorityQueue_Station.put((dfSt.loc["Finish"].iloc[i-1] + dfSt.loc[node].iloc[i-1] + distance , 
                                             [dfSt.loc[current[0]].index[i-1] ,dfSt.loc[current[0]].iloc[i-1]]))
            if priorityQueue_Station.empty() == True:
                # current = priorityQueue.get()[1] #Backtracking
                # path.append(current[0])
                # distance += current[1]
                # fuel -= current[1]
                continue
            else:
                current = priorityQueue_Station.get()[1]
                distance += current[1]
                fuel = fuel_capacity
                path.append(current[0])
                if Intervals.index(interval) != 0:
                    dfSt = DropStationDataFrame(dfSt, Intervals[Intervals.index(interval) - 1])
                priorityQueue_Station = queue.PriorityQueue()
                current, path, distance, fuel = StationPath(current[0], path, interval, dfSt, AllData, fuel_capacity, distance)
                # if Intervals.index(interval) != 0:
                #     dfSt = DropStationDataFrame(dfSt, Intervals[Intervals.index(interval) - 1])
                return current, path, distance, fuel
        else:
            distance += current[1]
            fuel -= current[1]
            path.append(current[0])
            if Intervals.index(interval) != 0:
                dfSt = DropStationDataFrame(dfSt, Intervals[Intervals.index(interval) - 1])
            return current, path, distance, fuel


def StationPath(node, path, interval, df, AllData, fuel, distance):
    priorityQueue = queue.PriorityQueue()
    priorityQueue_Station = queue.PriorityQueue()
    for i in range (interval_norm(interval)):
        priorityQueue.put((AllData["Heuristic"].iloc[i+interval[0]] + df[node].iloc[i] + distance,
                          [df[node].index[i], df[node].iloc[i]]))
    while priorityQueue.empty() == False:
        current = priorityQueue.get()[1]
        if current[1] > fuel:
            for i in range (1, interval_norm(intervalS) + 1):
                if dfSt[f"Station{i}"].loc[current[0]] > fuel:
                    continue
                else:
                    priorityQueue_Station.put((dfSt.loc["Finish"].iloc[i-1] + dfSt[node].iloc[i-1] + distance , 
                                             [dfSt.loc[current[0]].index[i-1] ,dfSt.loc[current[0]].iloc[i-1]]))
                
                
                    
            if priorityQueue_Station.empty() == True:
                # current = priorityQueue.get()[1] #Backtracking
                # path.append(current[0])
                # distance += current[1]
                # fuel -= current[1]
                continue
            else:
                current = priorityQueue_Station.get()[1]
                distance += current[1]
                fuel = fuel_capacity
                path.append(current[0])
                priorityQueue_Station = queue.PriorityQueue()
                current, path, distance, fuel = StationPath(current[0], path, interval, dfSt, AllData, fuel_capacity, distance)
                return current, path, distance, fuel
            
        else:
            distance += current[1]
            fuel -= current[1]
            path.append(current[0])
            return current, path, distance, fuel
        

def Astar():
    # current = ["Start", 0]
    # path = ["Start"]
    # distance = 0
    # fuel = fuel_capacity

    global current, path, distance, fuel, AllData
    for k in range (len(Intervals)):
        # try:
        #     current, path, distance, fuel = PointPath(path[-1], path, Intervals[k], DataFrame[k], AllData, fuel, distance)
        # except:
        #     print("bước cuối")
        #     break

        current, path, distance, fuel = PointPath(path[-1], path, Intervals[k], DataFrame[k], AllData, fuel, distance)
        

    return path, distance



current = ["Start", 0]
distance = 0
path = ["Start"]
fuel_capacity = 150
fuel = fuel_capacity

print(Astar())




# dfS1.to_csv('dfS1.csv', index=True)  # index=False để không bao gồm cột chỉ mục trong tệp CSV
# df12.to_csv('df12.csv', index=True)
# # df23.to_csv('df23.csv', index=True)
# df34.to_csv('df34.csv', index=True)
# df4F.to_csv('df4F.csv', index=True)