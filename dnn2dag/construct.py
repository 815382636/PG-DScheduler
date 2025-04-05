from node import Node
from job_dag import JobDAG
import random
import numpy as np
from karger import karger_min_cut

# 5
def construct(node_list, wall_time, np_random, source_device, job_idx):

    nodes = []
    nodes_dic = {}

    for i,v in enumerate(node_list):
        name = v[0]
        name = name[:name.index("(")].strip()
        size = v[1]
        output_size = 0.00001
        for j in size:
            if isinstance(j, int):
                output_size *= j
        output_size = round(output_size, 8)
        params = v[2] // 10000000 + random.randint(1, 10)
        if len(v) > 3:
            # 存在并行
            front = v[3]
        else:
            # 串行
            if i == 0:
                front = []
            else:
                front = [node_list[i - 1][0]]
        # print(name, "  |  ", output_size, "  |  ", params, "  |  ", front)

        node = Node(i, name, wall_time, np_random, params, 0, output_size)
        nodes.append(node)
        nodes_dic[name] = node

    adj_mat_dic = {t: [] for t in range(len(nodes))}
    for i,v in enumerate(node_list):
        if i > 0:
            if len(v) <= 3:
                adj_mat_dic[i - 1].append(i)
            else:
                # 存在并行
                for j in v[3]:
                    j_name = j[:j.index("[")].strip()
                    adj_mat_dic[nodes_dic[j_name].idx].append(i)

    # adj_mat = np.array(adj_mat)
    nodes, adj_mat_dic = karger_min_cut(nodes, adj_mat_dic)

    key_index = { v:i for i,v in enumerate(adj_mat_dic.keys())}
    adj_mat = [[ 0 for _ in range(len(key_index)) ] for t in range(len(key_index))]
    for k,v in adj_mat_dic.items():
        for next in v:
            adj_mat[key_index[k]][key_index[next]] = 1
    # print("邻接矩阵：", adj_mat)
    for node in nodes:
        node.idx = key_index[node.idx]

    for il, l in enumerate(adj_mat):
        for ir, r in enumerate(l):
            if r == 1:
                nodes[il].child_nodes.append(nodes[ir])
                nodes[ir].parent_nodes.append(nodes[il])
                nodes[ir].input_size = nodes[il].output_size

    roots = []

    # initialize descendant nodes
    for node in nodes:
        if len(node.parent_nodes) == 0:  # root
            get_descendant_data(node)
            roots.append(node)

    # adj_mat = np.array(adj_mat)
    # generate DAG
    # job_dag = JobDAG(nodes, adj_mat,
    #     "job_"+ str(job_idx), source_device, job_idx)
    
    job_dag = {"adj_mat": adj_mat}

    nodes_str =[]
    for i in nodes:
        nodes_str.append(i.to_dict())
    job_dag["nodes"] = nodes_str

    return job_dag

def get_descendant_data(node):
    if len(node.child_nodes) == 0:
        return []
    elif len(node.descendant_nodes) > 0:
        return node.descendant_nodes
    else:
        for child in node.child_nodes:
            child_descendant_nodes = get_descendant_data(child)
            for dn in child_descendant_nodes:
                if dn not in node.descendant_nodes:  # remove dual path duplicates
                    node.descendant_nodes.append(dn)
            if child not in node.descendant_nodes:
                node.descendant_nodes.append(child)
    return node.descendant_nodes
