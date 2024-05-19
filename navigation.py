import numpy as np
import time
from config import NEIGHBOR_POINTS, BIDIRECTIONAL_EDGES, SINGLE_DIRECTION_EDGES, NODE_COUNT
class DijkstraPathPlanning:
    def pathPlanner(self):
        n = NODE_COUNT
        # 初始化图
        map_matrix = np.full((n+1, n+1), np.inf)
        np.fill_diagonal(map_matrix, 0)
        
        # 更新图的双向路径和单向路径
        for i, j, d in BIDIRECTIONAL_EDGES :
            map_matrix[i][j] = d
            map_matrix[j][i] = d
        for i, j, d in SINGLE_DIRECTION_EDGES:
            map_matrix[i][j] = d
        
        # 随机选择两对点以生成点0和点12，保证这些点不直接连接到点7
        np.random.seed(int(time.time()))
        available_edges = [(i, j) for i, j, d in (BIDIRECTIONAL_EDGES + SINGLE_DIRECTION_EDGES) if 7 not in [i, j]]
        

        # 随机选择两条边用于放置点0和点12
        edge_for_0 = available_edges[np.random.randint(len(available_edges))]
        edge_for_12 = available_edges[np.random.randint(len(available_edges))]
        # edge_for_0 = [10, 9]
        # edge_for_12 = [8, 11]
        
        while edge_for_0 == edge_for_12:  # 确保两条边不相同
            edge_for_12 = available_edges[np.random.randint(len(available_edges))]
        
        # 分配距离
        dist_0_1 = np.random.randint(1, map_matrix[edge_for_0[0], edge_for_0[1]])
        dist_0_2 = map_matrix[edge_for_0[0], edge_for_0[1]] - dist_0_1
        dist_12_1 = np.random.randint(1, map_matrix[edge_for_12[0], edge_for_12[1]])
        dist_12_2 = map_matrix[edge_for_12[0], edge_for_12[1]] - dist_12_1
        
        # 更新矩阵以包括点0和点12的连接
        if ([edge_for_0[0],edge_for_0[1]] in NEIGHBOR_POINTS):
            map_matrix[edge_for_0[0], 0] = dist_0_1
            map_matrix[0, edge_for_0[0]] = dist_0_1
            map_matrix[edge_for_0[1], 0] = dist_0_2
            map_matrix[0, edge_for_0[1]] = dist_0_2
            map_matrix[edge_for_12[0], 12] = dist_12_1
            map_matrix[12, edge_for_12[0]] = dist_12_1
            map_matrix[edge_for_12[1], 12] = dist_12_2
            map_matrix[12, edge_for_12[1]] = dist_12_2
            map_matrix[edge_for_12[0], edge_for_12[1]] = np.inf
            map_matrix[edge_for_12[1], edge_for_12[0]] = np.inf
            
        if ([edge_for_0[0],edge_for_0[1]] not in NEIGHBOR_POINTS):
            map_matrix[edge_for_0[0], 0] = dist_0_1
            map_matrix[0, edge_for_0[1]] = dist_0_2
            map_matrix[edge_for_12[0], 12] = dist_12_1
            map_matrix[12, edge_for_12[1]] = dist_12_2
            map_matrix[edge_for_12[0], edge_for_12[1]] = np.inf
            map_matrix[edge_for_12[1], edge_for_12[0]] = np.inf
                
        
        def dijkstra(graph, start, end):
            n = len(graph)
            shortest_path = {vertex: np.inf for vertex in range(n)}
            previous_nodes = {vertex: None for vertex in range(n)}
            shortest_path[start] = 0
            unvisited_nodes = set(range(n))
            
            while unvisited_nodes:
                current_node = min(unvisited_nodes, key=lambda vertex: shortest_path[vertex])
                unvisited_nodes.remove(current_node)
                
                if shortest_path[current_node] == np.inf or current_node == end:
                    break
                
                for neighbor, distance in enumerate(graph[current_node]):
                    if distance + shortest_path[current_node] < shortest_path[neighbor]:
                        shortest_path[neighbor] = distance + shortest_path[current_node]
                        previous_nodes[neighbor] = current_node
            
            path, current_node = [], end
            while current_node is not None:
                path.insert(0, current_node)
                current_node = previous_nodes[current_node]
            
            return path
        
        # 计算从点6到0，0到12，12到6的最短路径
        path_6_to_0 = dijkstra(map_matrix, 6, 0)
        path_0_to_12 = dijkstra(map_matrix, 0, 12)
        path_12_to_6 = dijkstra(map_matrix, 12, 6)
        
        # 合并路径
        complete_path = path_6_to_0 + path_0_to_12[1:] + path_12_to_6[1:]
        
        complete_path, edge_for_0, dist_0_1, dist_0_2, edge_for_12, dist_12_1, dist_12_2
        
        print("抓取货物点：距离" + str(edge_for_0) + " " + str(dist_0_1) +"和" +str(dist_0_2)+ "cm")
        print("送达点：距离" + str(edge_for_12) + " " + str(dist_12_1) +"和" +str(dist_12_2)+ "cm")
        
        print(complete_path)
        result = {
            'path': complete_path,
            'pickup_point': edge_for_0,
            'dist_0_1': dist_0_1,
            'dist_0_2': dist_0_2,
            'destination': edge_for_12,
            'dist_12_1': dist_12_1,
            'dist_12_2': dist_12_2
        }
        
        return result