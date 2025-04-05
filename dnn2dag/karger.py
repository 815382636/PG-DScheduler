import random
import copy

# 随机选择一个边的端点
def random_edge(graph):
    u = random.choice(list(graph.keys()))
    while(len(graph[u]) == 0):
        u = random.choice(list(graph.keys()))
    v = random.choice(graph[u])
    return u, v

# 合并两个顶点
def contract(test_nodes, test_adj, u, v):
    # print(test_adj, "-", u, "-", v)
    ind = 0
    for items in test_adj.values():
         for item in items:
              if v == item:
                   ind += 1
    if ind >= 2:
         return

    u_node = None
    v_node = None
    for i in test_nodes:
        if i.idx == u:
             u_node = i
        if i.idx == v:
             v_node = i

    u_node.workload += v_node.workload

    # 将v的所有边移到u上
    for neighbor in test_adj[v]:
            if neighbor not in test_adj[u]:
                test_adj[u].append(neighbor)
    del test_adj[v]
    test_adj[u].remove(v)

    # 删除v
    test_nodes.remove(v_node)

# Karger算法
def karger_min_cut(nodes, adj_mat_dic):
    # print(adj_mat_dic)
    
    result = 999999999
    new_nodes = []
    new_adj = {}

    for i in range(len(nodes) * 10):
        test_nodes = copy.deepcopy(nodes)
        test_adj = copy.deepcopy(adj_mat_dic)
        # return_num = random.randint(5, 8)
        return_num = random.randint(10, 13)

        while(len(test_nodes) > return_num):
            u, v = random_edge(test_adj)
            contract(test_nodes, test_adj, u, v)
            # print(len(test_nodes))

        # print("---------------------")
        # print(test_adj)
        # print("---------------------")
 
        mid_result = 0
        for i in test_nodes:
             mid_result += i.output_size

        if mid_result < result:
             new_nodes = test_nodes
             new_adj = test_adj
    # print("---------------------")
    # print(new_adj)
    # print("---------------------")   
    
    return new_nodes, new_adj
