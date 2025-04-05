from spark_env.node import Node
from spark_env.job_dag import JobDAG
from param import *
import random
import numpy as np

# 5
def construct(job, wall_time, np_random, source_device, job_idx):

    adj_mat = job["adj_mat"]
    nodes_str = job["nodes"]

    nodes = []
    for i in nodes_str:
        node = Node(i["idx"], i["name"], wall_time, np_random, i["workload"], i["input_size"], i["output_size"])
        nodes.append(node)

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

    adj_mat = np.array(adj_mat)
    # generate DAG
    job_dag = JobDAG(nodes, adj_mat,
        "job_"+ str(job_idx), args.enb_num + source_device, job_idx)

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
