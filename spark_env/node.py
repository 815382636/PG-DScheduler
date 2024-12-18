import numpy as np
from param import *
from utils import OrderedSet


class Node(object):
    def __init__(self, idx, wall_time, np_random, workload):
        self.idx = idx
        # self.tasks = tasks
        self.wall_time = wall_time
        self.np_random = np_random
        self.occupied = False
        self.enb = None
        self.node_finish_time = np.inf
        self.node_start_time = np.inf
        self.node_finished = False


        # self.task_duration = task_duration
        self.workload = workload
        self.input_size = 0.00000001
        self.output_size = 0.00000001

        self.tran_loss = -1
        self.waste_loss = -1
        self.exec_loss = -1

        # uninitialized
        self.parent_nodes = []
        self.child_nodes = []
        self.descendant_nodes = []
        self.job_dag = None

    def is_schedulable(self):
        parents_occupied = True
        for parent_node in self.parent_nodes:
            if not parent_node.occupied:
                parents_occupied = False
                break
        if parents_occupied is True and self.occupied is not True:
            return True
        else:
            return False

    def reset(self):
        self.node_finish_time = np.inf
        self.occupied = False
        self.enb = None



