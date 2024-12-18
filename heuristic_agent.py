#coding=utf-8
import numpy as np
from param import *

from spark_env.enb import Enb
from spark_env.job_dag import JobDAG
from spark_env.mobile_device import MobileDevice
from spark_env.node import Node


def get_node_est_by_parents(node, server_id, enb_adj, curr_time):
    maxi = curr_time
    for parent in node.parent_nodes:
        if parent.enb.idx != server_id:
            if (max(curr_time, parent.node_finish_time) + (parent.output_size / enb_adj[server_id][parent.enb.idx])) > maxi:
                maxi = max(curr_time, parent.node_finish_time) + (parent.output_size / enb_adj[server_id][parent.enb.idx])
        else:
            if parent.node_finish_time > maxi:
                maxi = parent.node_finish_time
    # if len(node.parent_nodes) == 0 and server_id != node.job_dag.source_id:
    #     trans_time = node.input_size / enb_adj[server_id][node.job_dag.source_id]
    #     maxi = trans_time + curr_time
    return maxi

class DynamicPartitionAgent():
    # dynamically partition the cluster resource
    # scheduling complexity: O(num_nodes * num_executors)
    def __init__(self, flag):
        self.flag = flag
        pass

    def get_action(self, obs, scheme):

        # parse observation
        job_dags, enb_adj, num_avail_position, \
        frontier_nodes, action_map, servers, curr_time = obs
        enb_valid = np.zeros([1, len(servers)])

        total_num_nodes = int(np.sum(job_dag.num_nodes for job_dag in job_dags))

        for server in servers:
            if not server.occupied or not server.node_wait.full():
                enb_valid[0, server.idx] = 1

        # new_time_list = []
        # obj_list = []
        # while True:
        #     if len(timeline_save) == 0:
        #         break
        #     new_time, obj = timeline_save.pop()
        #     new_time_list.append(new_time)
        #     obj_list.append(obj)
        #     if curr_time + args.enb_delay < new_time:
        #         break
        #     if (isinstance(obj, Enb) or isinstance(obj, MobileDevice)) and not obj.node_wait.full():
        #         enb_valid[0, obj.idx] = 1
        #
        # for idx in range(len(new_time_list)):
        #     timeline_save.push(new_time_list[idx], obj_list[idx])

        # 选node
        node = None
        # SJF
        if self.flag == 0:
            job_exclude = []
            while node is None:
                min_node_num = 9999
                job_temp = None
                for job in job_dags:
                    node_num = job.get_frontier_nodes_workload()
                    if min_node_num > node_num > 0 and job not in job_exclude:
                        job_temp = job
                        min_node_num = node_num
                for next_node in job_temp.nodes:
                    if len(next_node.child_nodes) == 0 or len(next_node.parent_nodes) == 0:
                        if enb_valid[0, next_node.job_dag.source_id] == 1 and next_node.is_schedulable() is True:
                            return next_node, next_node.job_dag.source_id,  len(frontier_nodes), total_num_nodes
                    elif next_node.is_schedulable() is True:
                        for j in range(len(servers)):
                            if enb_valid[0, j] == 1 and (j < args.enb_num or j == next_node.job_dag.source_id):
                                node = next_node
                                break
                    if node is not None:
                        break
                job_exclude.append(job_temp)

            assert node is not None
            #
        # FIFO
        # node = None
        elif self.flag == 2 or self.flag == 5 or self.flag == 6:
            while node is None:
                for job in job_dags:
                    for next_node in job.nodes:
                        if len(next_node.child_nodes) == 0 or len(next_node.parent_nodes) == 0:
                            if enb_valid[0, next_node.job_dag.source_id] == 1 and next_node.is_schedulable() is True:
                                return next_node, next_node.job_dag.source_id, len(frontier_nodes), total_num_nodes
                        elif next_node.is_schedulable() is True:
                            if self.flag == 2:
                                for j in range(len(servers)):
                                    if enb_valid[0, j] == 1 and (j < args.enb_num or j == next_node.job_dag.source_id):
                                        node = next_node
                                        break
                            elif self.flag == 5:
                                for j in range(args.enb_num):
                                    if enb_valid[0, j] == 1:
                                        node = next_node
                                        break
                            elif self.flag == 6:
                                if enb_valid[0, next_node.job_dag.source_id] == 1:
                                    node = next_node
                        if node is not None:
                            break
                    if node is not None:
                        break


        #avg
        elif self.flag == 1:
            job_server_map = {}

            for job in job_dags:
                job_server_map[job] = 0
            for server in servers:
                if server.node is not None:
                    job_server_map[server.node.job_dag] += 1
                for node_l in server.node_list:
                    job_server_map[node_l.job_dag] += 1

            job_exclude = []
            while node is None:
                minn = 9999
                for job in job_dags:
                    if job_server_map[job] < minn and job.get_frontier_nodes_length() > 0 and job not in job_exclude:
                        minn = job_server_map[job]
                        job_target = job

                for next_node in job_target.nodes:
                    if len(next_node.child_nodes) == 0 or len(next_node.parent_nodes) == 0:
                        if enb_valid[0, next_node.job_dag.source_id] == 1 and next_node.is_schedulable() is True:
                            return next_node, next_node.job_dag.source_id,  len(frontier_nodes), total_num_nodes
                    elif next_node.is_schedulable() is True:
                        for j in range(len(servers)):
                            if enb_valid[0, j] == 1 and (j < args.enb_num or j == next_node.job_dag.source_id):
                                node = next_node
                                break
                    if node is not None:
                        break
                job_exclude.append(job_target)
        #
        #
        # Random
        elif self.flag == 3 or self.flag == 4:
            len_nodes = len(frontier_nodes)
            idx = np.random.randint(len_nodes)
            node = frontier_nodes.to_list()[idx]
            if len(node.child_nodes) == 0 or len(node.parent_nodes) == 0:
                if node.is_schedulable() is True:
                    assert enb_valid[0, node.job_dag.source_id] == 1
                    return node, node.job_dag.source_id, len(frontier_nodes), total_num_nodes

        assert node is not None

        # greedy
        # waste_time_min = 100000000
        # target_server = -1
        # flag_temp = False
        # if len(node.parent_nodes) != 0 and len(node.child_nodes) != 0:
        #     for j in range(args.enb_num + 1):
        #         if j < args.enb_num and enb_valid[0, j] == 1:
        #             est_by_server = max(servers[j].avail_time, curr_time)
        #             est_by_parents = get_node_est_by_parents(node, j, enb_adj, curr_time)
        #             flag_temp = True
        #         elif j == args.enb_num and enb_valid[0, node.job_dag.source_id] == 1:
        #             est_by_server = max(servers[node.job_dag.source_id].avail_time, curr_time)
        #             est_by_parents = get_node_est_by_parents(node, node.job_dag.source_id, enb_adj, curr_time)
        #             flag_temp = True
        #         if flag_temp:
        #             waste_time = abs(est_by_server - est_by_parents)
        #             if waste_time < waste_time_min:
        #                 waste_time_min = waste_time
        #                 target_server = j
        #             flag_temp = False
        #     assert target_server != -1
        #     assert waste_time_min >= 0
        #
        # assert target_server != -1
        # if target_server == args.enb_num:
        #     target_server = node.job_dag.source_id

        # if len(node.child_nodes) == 0:
        #     enb_idx = node.job_dag.source_id

        min = 999999999999
        target_server = -1

        if self.flag != 5 and self.flag != 6:
            for j in range(len(servers)):
                if enb_valid[0, j] == 1 and (j < args.enb_num or j == node.job_dag.source_id):
                    est = max(servers[j].avail_time, get_node_est_by_parents(node, j, enb_adj, curr_time), curr_time) + node.workload / servers[j].computing_power
                    if est < min:
                        min = est
                        target_server = j

            if self.flag == 3:
                server_list = []
                for j in range(len(servers)):
                    if enb_valid[0, j] == 1 and (j < args.enb_num or j == node.job_dag.source_id):
                        server_list.append(j)
                target_server = np.random.choice(server_list)
        else:
            if self.flag == 5:
                if len(node.parent_nodes) != 0 and len(node.child_nodes) != 0:
                    for j in range(args.enb_num):
                        if enb_valid[0, j] == 1:
                            est = max(servers[j].avail_time, get_node_est_by_parents(node, j, enb_adj, curr_time),
                                      curr_time) + node.workload / servers[j].computing_power
                            if est < min:
                                min = est
                                target_server = j
            elif self.flag == 6:
                assert enb_valid[0, node.job_dag.source_id] == 1
                target_server = node.job_dag.source_id

        assert target_server != -1

        return node, target_server, len(frontier_nodes), total_num_nodes