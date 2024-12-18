# coding=utf-8
import numpy as np
import copy
from collections import OrderedDict
from param import *
from utils import *
from actor_agent import get_node_est_by_parents
from spark_env.action_map import compute_act_map
from spark_env.reward_calculator import RewardCalculator
from spark_env.job_generator import generate_jobs
from spark_env.wall_time import WallTime
from spark_env.timeline import Timeline
from spark_env.job_dag import JobDAG
from spark_env.enb import Enb
from spark_env.mobile_device import MobileDevice


class Environment(object):
    def __init__(self, test_flag=0):

        # isolated random number generator
        self.np_random = np.random.RandomState()
        self.np_random_workload = np.random.RandomState()

        # global timer
        self.wall_time = WallTime()

        # uses priority queue
        self.timeline = Timeline()

        self.servers_num = args.enb_num + args.mobile_device_num

        # 各个服务器
        self.servers = []
        for server_id in range(args.enb_num):
            enb = Enb(server_id, 2.5 * args.enb_cores_nums)
            self.servers.append(enb)

        ############################################
        for server_id in range(args.mobile_device_num):
            id = server_id + args.enb_num
            mobile_device = MobileDevice(id, 2.0 * args.md_cores_nums)
            self.servers.append(mobile_device)

        # 服务器之间带宽
        self.enb_adj = np.zeros([self.servers_num, self.servers_num])

        # for computing reward at each step
        self.reward_calculator = RewardCalculator()

        self.test_flag = test_flag

    # 找出所有可调度的节点，即父节点均已被调度，自身未调度，并且有可分配的服务器
    def get_frontier_nodes(self):
        # frontier nodes := unsaturated nodes with all parent nodes saturated
        frontier_nodes = OrderedSet()
        for job_dag in self.job_dags:
            for node in job_dag.nodes:
                if not node.occupied:
                    parents_occupied = True
                    for parent_node in node.parent_nodes:
                        if not parent_node.occupied:
                            parents_occupied = False
                            break
                    if parents_occupied:
                        if len(node.child_nodes) == 0 or len(node.parent_nodes) == 0:
                            if self.get_avail_server()[0, node.job_dag.source_id] == 1:
                                frontier_nodes.add(node)
                        else:
                            # frontier_nodes.add(node)
                            enb_valid = self.get_avail_server()
                            if self.test_flag == 0:
                                for i in range(args.enb_num + args.mobile_device_num):
                                    if (i < args.enb_num or i == node.job_dag.source_id) and enb_valid[0, i] == 1:
                                        frontier_nodes.add(node)
                                        break
                            elif self.test_flag == 1:
                                if enb_valid[0, node.job_dag.source_id] == 1:
                                    frontier_nodes.add(node)
                                    break
                            elif self.test_flag == 2:
                                for i in range(args.enb_num):
                                    if enb_valid[0, i] == 1:
                                        frontier_nodes.add(node)
                                        break

        return frontier_nodes


    def observe(self):
        return self.job_dags, self.enb_adj, self.get_num_avail_position(), \
               self.get_frontier_nodes(), self.action_map, self.servers, self.wall_time.curr_time

    # 确定是否有服务器空闲,有则触发调度事件
    def get_num_avail_position(self):
        num = 0
        server_state = self.get_server_state()
        for server in self.servers:
            if server.idx < args.enb_num:
                if server_state[server.idx]:
                    if not server.occupied:
                        num += 1
            else:
                if server_state[server.idx]:
                    if not server.occupied:
                        num += 1
        return num

    # 确定剩余可调度的节点所属的DAG里,是否有从对应移动设备产生的
    def get_server_state(self):
        state = []
        for i in range(args.enb_num + args.mobile_device_num):
            state.append(False)
        for job_dag in self.job_dags:
            for node in job_dag.nodes:
                if node.is_schedulable():
                    state[node.job_dag.source_id] = True

        for job_dag in self.job_dags:
            for node in job_dag.nodes:
                if node.is_schedulable() and len(node.parent_nodes) != 0 and len(node.child_nodes) != 0:
                    for i in range(args.enb_num):
                        state[i] = True
                    break
            if state[0]:
                break
        return state

    # 是否可以给对应服务器分配节点
    def get_avail_server(self):

        enb_valid = np.zeros([1, len(self.servers)])
        for server in self.servers:
            if not server.occupied or not server.node_wait.full():
                enb_valid[0, server.idx] = 1

        return enb_valid

    # 给服务器分配节点
    def assign_server(self, node, server_id):

        node.enb = self.servers[server_id]
        node.occupied = True
        # 所有父节点完成传输时刻
        est_by_parents = get_node_est_by_parents(node, server_id, self.enb_adj, self.wall_time.curr_time)
        # 开始时间
        est = max(self.wall_time.curr_time, self.servers[server_id].avail_time, est_by_parents)
        # 传输损失相关
        node.tran_loss = (abs(max(self.servers[server_id].avail_time, self.wall_time.curr_time) - est_by_parents)) / 20
        node.waste_loss = abs(max(self.servers[server_id].avail_time, self.wall_time.curr_time) - est_by_parents) / 20
        node.exec_loss = node.workload / self.servers[server_id].computing_power / 20
        # 记录开始时间与完成时间
        end_time = est + node.workload / self.servers[server_id].computing_power
        node.node_start_time = est
        node.node_finish_time = end_time

        # 服务器空闲
        if not self.servers[server_id].occupied:
            assert self.servers[server_id].avail_time <= self.wall_time.curr_time
            self.servers[server_id].node = node
            self.servers[server_id].job = node.job_dag
            self.servers[server_id].occupied = True
            self.timeline.push(end_time, self.servers[server_id])
        # 服务器不空，进队列
        else:
            self.servers[server_id].node_wait.put(node)
            self.servers[server_id].node_list.append(node)
        self.servers[server_id].avail_time = node.node_finish_time

    def step(self, next_node, server_id, node_idx, exp):

        if server_id == args.enb_num:
            server_id = next_node.job_dag.source_id

        assert next_node is not None
        self.assign_server(next_node, server_id)
        if len(next_node.child_nodes) == 0 or len(next_node.parent_nodes) == 0:
            assert server_id == next_node.job_dag.source_id

        total_num_nodes = int(np.sum(job_dag.num_nodes for job_dag in self.job_dags))
        tran_loss_vec = np.zeros([1, total_num_nodes])
        waste_loss_vec = np.zeros([1, total_num_nodes])
        exec_loss_vec = np.zeros([1, total_num_nodes])
        assert next_node.tran_loss >= 0
        if len(next_node.child_nodes) != 0 and len(next_node.parent_nodes) != 0:
            tran_loss_vec[0, node_idx] = next_node.tran_loss
            waste_loss_vec[0, node_idx] = next_node.waste_loss
            exec_loss_vec[0, node_idx] = next_node.exec_loss
        else:
            tran_loss_vec[0, node_idx] = -1
        exp['tran_loss'].append(tran_loss_vec)
        exp['waste_loss'].append(waste_loss_vec)
        exp['exec_loss'].append(exec_loss_vec)

        # 判断是否满足动作触发条件
        while (self.get_num_avail_position() == 0 or len(self.get_frontier_nodes()) == 0) and len(self.timeline) != 0:

            new_time, obj = self.timeline.pop()
            assert new_time >= self.wall_time.curr_time
            self.wall_time.update_time(new_time)

            if isinstance(obj, JobDAG):  # new job arrival event
                job_dag = obj
                # job should be arrived at the first time
                assert not job_dag.arrived
                job_dag.arrived = True
                # inform agent about job arrival when stream is enabled
                self.job_dags.add(job_dag)
                self.action_map = compute_act_map(self.job_dags)

            elif isinstance(obj, Enb) or isinstance(obj, MobileDevice):  # node完成
                node = obj.node
                node.node_finished = True
                node.job_dag.num_nodes_done += 1

                if node.job_dag.num_nodes_done == node.job_dag.num_nodes:
                    assert not node.job_dag.completed  # only complete once
                    node.job_dag.completed = True
                    node.job_dag.completion_time = self.wall_time.curr_time
                    self.remove_job(node.job_dag)

                if not obj.node_wait.empty():
                    node_next = obj.node_wait.get()
                    obj.node_list.remove(node_next)
                    self.timeline.push(node_next.node_finish_time, obj)
                    obj.node = node_next
                    obj.job = node_next.job_dag
                else:
                    obj.occupied = False
                    obj.node = None
                    obj.job = None

            else:
                print("illegal event type")
                exit(1)

        # compute reward
        reward = self.reward_calculator.get_reward(
            self.job_dags, self.wall_time.curr_time)

        # no more decision to make, jobs all done or time is up
        done = (len(self.timeline) == 0 and len(self.get_frontier_nodes()) == 0)

        if done:
            assert len(self.job_dags) == 0

        return self.observe(), reward, done

    def remove_job(self, job_dag):
        self.job_dags.remove(job_dag)
        self.finished_job_dags.add(job_dag)
        self.action_map = compute_act_map(self.job_dags)

    def reset(self, test_flag=0):
        self.wall_time.reset()
        self.timeline.reset()
        self.reward_calculator.reset()
        self.finished_job_dags = OrderedSet()
        self.job_dags = generate_jobs(
            self.np_random, self.np_random_workload, self.timeline, self.wall_time)
        # map action to dag_idx and node_idx
        self.action_map = compute_act_map(self.job_dags)
        self.reset_enb_adj()
        # self.exec_to_schedule = OrderedSet(self.executors)
        self.reset_servers()
        self.test_flag = test_flag

    def seed(self, seed):
        self.np_random.seed(seed)
        self.np_random_workload.seed(43)


    def reset_servers(self):

        self.servers.clear()

        for server_id in range(args.enb_num):
            enb = Enb(server_id, 2.5 * args.enb_cores_nums)
            self.servers.append(enb)

        for server_id in range(args.mobile_device_num):
            id = server_id + args.enb_num
            mobile_device = MobileDevice(id, 2.0 * args.md_cores_nums)
            self.servers.append(mobile_device)

    def reset_enb_adj(self):
        for i in range(self.servers_num):
            for j in range(self.servers_num):
                if i < j:
                    self.enb_adj[i][j] = self.np_random_workload.randint(10, 100)
                    self.enb_adj[j][i] = self.enb_adj[i][j]
