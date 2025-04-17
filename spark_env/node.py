import numpy as np
from param import *
from utils import OrderedSet
import random


class Node(object):
    def __init__(self, idx, name, wall_time, np_random, params, input_size=0.00000001, output_size = 0.00000001):
        self.idx = idx
        self.name = name
        # self.tasks = tasks
        self.wall_time = wall_time
        self.np_random = np_random
        self.occupied = False
        self.enb = None
        self.node_finish_time = np.inf
        self.node_start_time = np.inf
        self.node_finished = False

        # self.task_duration = task_duration
        self.workload = params
        self.input_size = input_size
        self.output_size = output_size

        self.back_workload = random.randint(self.workload / 2, self.workload)
        self.back_input_size = output_size
        self.back_output_size = input_size

        self.tran_loss = -1
        self.waste_loss = -1
        self.exec_loss = -1

        # uninitialized
        self.parent_nodes = []
        self.child_nodes = []
        self.descendant_nodes = []
        self.job_dag = None

        # 反向传播标识
        self.back_sign = False

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



