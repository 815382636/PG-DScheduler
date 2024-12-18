import networkx as nx
import numpy as np
from collections import OrderedDict
from utils import OrderedSet
from param import *

class JobDAG(object):
    def __init__(self, nodes, adj_mat, name, source_id, job_idx):
        # nodes: list of N nodes
        # adj_mat: N by N 0-1 adjacency matrix, e_ij = 1 -> edge from i to j
        assert len(nodes) == adj_mat.shape[0]
        assert adj_mat.shape[0] == adj_mat.shape[1]

        self.name = name
        self.job_idx = job_idx

        self.nodes = nodes
        self.adj_mat = adj_mat
        self.source_id = source_id

        self.num_nodes = len(self.nodes)
        self.num_nodes_done = 0

        # the computation graph needs to be a DAG
        assert is_dag(self.num_nodes, self.adj_mat)
        assert args.enb_num + args.mobile_device_num > source_id >= args.enb_num

        # assign job dag to node
        self.assign_job_dag_to_node()

        # dag is arrived
        self.arrived = False

        # dag is completed
        self.completed = False

        # dag start ime
        self.start_time = None

        # dag completion time
        self.completion_time = np.inf


    def assign_job_dag_to_node(self):
        for node in self.nodes:
            node.job_dag = self

    def get_frontier_nodes_workload(self):
        sum = 0
        for node in self.nodes:
            if node.occupied is not True:
                sum += node.workload
        return sum

    def reset(self):
        for node in self.nodes:
            node.reset()
        self.num_nodes_done = 0
        self.arrived = False
        self.completed = False
        self.completion_time = np.inf
        self.source_id = None


def is_dag(num_nodes, adj_mat):
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_mat[i, j] == 1:
                G.add_edge(i, j)
    return nx.is_directed_acyclic_graph(G)
