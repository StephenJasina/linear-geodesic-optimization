import networkx as nx
import pandas as pd
import numpy as np


def generating_tessalation(path_to_graph,name_of_output,name_of_output_cities,precision=0.05,tessalation_grid=0.02):
    '''
    :param path_to_graph: string to where the graph is located
    :param name_of_output: string name of the output for the curvature csv
    :param name_of_output_cities: string name of the output to find the coordinates in the city in the 2D grid
    :param precision: float precision value
    :param tessalation_grid: float other type of precesion value
    :return: two csv files one is output_cities and the other one output
    '''
    edge = 0.5
    # x and y are the corresponding indices of a 2D mesh
    x, y = np.mgrid[-edge:edge:tessalation_grid, -edge:edge:tessalation_grid]
    # create the structure needed for evaluating pdfs
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    gridsize = x.shape[0]
    G = nx.read_graphml(path_to_graph)
    # print(G.nodes(data=True))
    x_graph = nx.get_node_attributes(G,'long')
    y_graph = nx.get_node_attributes(G,'lat')
    # print(x_graph,y_graph)
    max_x = max(x_graph.values())
    min_x = min(x_graph.values())
    max_y = max(y_graph.values())
    min_y = min(y_graph.values())
    for t in x_graph.keys():
        x_graph[t]= (x_graph[t] - min_x)/ (max_x - min_x) - 0.5
        y_graph[t]= -((y_graph[t]-min_y) / (max_y - min_y) - 0.5)
    city = nx.get_node_attributes(G,'city')
    nx.set_node_attributes(G,x_graph,'x')
    nx.set_node_attributes(G,y_graph,'y')
    curv = nx.get_edge_attributes(G,'ricciCurvature')
    # print(x_graph,max_x)
    coordinates = {}
    for t in pos:
        for s in t:
            coordinates[tuple(s)] = []
    #         number_of_edges_in_this_field[tuple(s)] = 0
    from tqdm import tqdm
    for i,t in enumerate(tqdm(G.edges())):
        number_of_edges_in_this_field = {}
        for var in np.arange(0,1,precision):
            line = ((x_graph[t[0]]-x_graph[t[1]])*var + x_graph[t[0]],(y_graph[t[0]]-y_graph[t[1]])*var + (y_graph[t[0]]))
            for s in coordinates.keys():
                if np.linalg.norm(np.array(s)-np.array(line),1)<precision and  not(s in number_of_edges_in_this_field.keys()):
                    number_of_edges_in_this_field[s] = 1
                    coordinates[s].append(curv[t])
        print(len(number_of_edges_in_this_field))
    new_x_graph = dict(zip(city.values(),x_graph.values()))
    new_y_graph = dict(zip(city.values(),y_graph.values()))
    print(len(city.values()),len(x_graph.values()),len(new_x_graph))
    df_inter = pd.Series(new_x_graph)
    df_inter = pd.DataFrame(df_inter,columns=['x'])
    df_inter['y'] = pd.Series(new_y_graph)
    df_inter.to_csv(name_of_output_cities)
    for t in coordinates.keys():
        try:
            if len(coordinates[t])==0:
                coordinates[t] = 0
            else:
                coordinates[t] = np.mean(coordinates[t])
        except:
            continue
    indexation_x = []
    indexation_y = []
    for s in x[:,0]:
        indexation_x.append(round(s,2))
    for s in x[:,0]:
        indexation_y.append(round(s,2))
    new_coordinates = {}
    for t in coordinates.keys():
        new_coordinates[round(t[0],2),round(t[1],2)] = coordinates[t]
    coordinates = new_coordinates
    df = pd.DataFrame([[0.0]*len(indexation_x)]*len(indexation_y),index=indexation_x,columns=indexation_y)

    for t in coordinates.keys():
        if t[0] in df.index and t[1] in df.columns:
            df[t[0]][t[1]] = coordinates[t]
            if coordinates[t]<0:
                print('success',df[t[0]][t[1]],coordinates[t])
    df.to_csv(name_of_output+'.csv')
    print(df.head())

''' 
This is a Python function that generates a tessellation of a graph in two dimensions.
It sets up a 2D mesh grid with dimensions determined by the edge and tessalation_grid parameters.
It reads in the input graph and extracts the node coordinates and edge curvature values from it.
It normalizes the node coordinates to be between -0.5 and 0.5.
It sets up an empty dictionary to store the coordinates of the tessellated grid.
It iterates over the edges of the graph and uses a precision value to determine how many points on each edge should be added to the tessellated grid.
For each point on an edge that should be added to the grid, the function increments the value stored at that point in the dictionary by the curvature value of the edge.
'''
def generating_tessalation_2(graphml_graph,precision=0.05,tessalation_grid=0.02):
    '''
    :param path_to_graph: string to where the graph is located
    :param name_of_output: string name of the output for the curvature csv
    :param name_of_output_cities: string name of the output to find the coordinates in the city in the 2D grid
    :param precision: float precision value
    :param tessalation_grid: float other type of precesion value
    :return: two csv files one is output_cities and the other one output
    '''
    edge = 0.5
    name_of_output = "output2"
    name_of_output_cities = "cities2.csv"
    # x and y are the corresponding indices of a 2D mesh
    x, y = np.mgrid[-edge:edge:tessalation_grid, -edge:edge:tessalation_grid]
    # create the structure needed for evaluating pdfs
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    print(pos)
    gridsize = x.shape[0]
    # G = nx.read_graphml(path_to_graph)
    G = graphml_graph
    # print("G.nodes")
    # print(G.nodes(data=True))
    x_graph = nx.get_node_attributes(G,'long')
    y_graph = nx.get_node_attributes(G,'lat')
    # print("x_graph, y_graph")
    # print(x_graph,y_graph)
    max_x = max(x_graph.values())
    min_x = min(x_graph.values())
    max_y = max(y_graph.values())
    min_y = min(y_graph.values())
    ####!!!!CHANGE
    for t in x_graph.keys():
        x_graph[t]= ((x_graph[t] - min_x)/ (max_x - min_x) - 0.5)
        y_graph[t]= -((y_graph[t]-min_y) / (max_y - min_y) - 0.5)
    city = nx.get_node_attributes(G,'city')
    nx.set_node_attributes(G,x_graph,'x')
    nx.set_node_attributes(G,y_graph,'y')
    curv = nx.get_edge_attributes(G,'ricciCurvature')
    # print("curv")
    # print(curv)
    # print("x_graph, max_x")
    # print(x_graph,max_x)
    coordinates = {}
    for t in pos:
        for s in t:
            coordinates[tuple(s)] = []
    #         number_of_edges_in_this_field[tuple(s)] = 0
    # print(coordinates)
    from tqdm import tqdm
    ### iterate through the edges and fill in the entries
    for i,t in enumerate(tqdm(G.edges())):
        number_of_edges_in_this_field = {}
        for var in np.arange(0,1,precision):

            line = ((x_graph[t[1]]-x_graph[t[0]])*var + x_graph[t[0]],(y_graph[t[1]]-y_graph[t[0]])*var + (y_graph[t[0]]))
            for s in coordinates.keys():
                # print(np.array(line))

                if np.linalg.norm(np.array(s)-np.array(line),1)<precision and  not(s in number_of_edges_in_this_field.keys()):
                    number_of_edges_in_this_field[s] = 1
                    # print("coords, curv, t")
                    # print(coordinates[s])
                    # print(curv, t)
                    #### This seems like a good place to update the code
                    coordinates[s].append(curv[t])
        print(len(number_of_edges_in_this_field))
    new_x_graph = dict(zip(city.values(),x_graph.values()))
    new_y_graph = dict(zip(city.values(),y_graph.values()))
    print(len(city.values()),len(x_graph.values()),len(new_x_graph))
    df_inter = pd.Series(new_x_graph)
    df_inter = pd.DataFrame(df_inter,columns=['x'])
    df_inter['y'] = pd.Series(new_y_graph)
    df_inter.to_csv(name_of_output_cities)
    ## DO SOMETHING ELSE
    for t in coordinates.keys():
        try:
            if len(coordinates[t])==0:
                coordinates[t] = 0
            else:
                if np.amin(coordinates[t]) < -0.8:
                    coordinates[t] = np.amin(coordinates[t])
                else:
                    coordinates[t] = np.mean(coordinates[t])

        except:
            continue
    indexation_x = []
    indexation_y = []
    for s in x[:,0]:
        indexation_x.append(round(s,2))
    for s in x[:,0]:
        indexation_y.append(round(s,2))
    new_coordinates = {}
    for t in coordinates.keys():
        new_coordinates[round(t[0],2),round(t[1],2)] = coordinates[t]
    coordinates = new_coordinates
    df = pd.DataFrame([[0.0]*len(indexation_x)]*len(indexation_y),index=indexation_x,columns=indexation_y)

    for t in coordinates.keys():
        if t[0] in df.index and t[1] in df.columns:
            df[t[0]][t[1]] = coordinates[t]
            # if coordinates[t]<0:
            #     print('success',df[t[0]][t[1]],coordinates[t])
    # print(df.to_csv())
    df.to_csv(name_of_output+'.csv')
    return df.to_csv()


if __name__ == '__main__':
    generating_tessalation('/Users/geode/Documents/Research Related/Internet of Space and Time /azure_cloud/Newazure_cloud70.graphml','tessalation_test.csv','city_test.csv',0.05,0.01)
