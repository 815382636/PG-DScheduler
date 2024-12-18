#coding=utf-8
import os

from heuristic_agent import DynamicPartitionAgent

os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from param import *
from utils import *
from spark_env.env import Environment
from average_reward import *
from compute_baselines import *
from compute_gradients import *
from actor_agent import ActorAgent, get_node_est_by_parents
from tf_logger import TFLogger
import copy
import matplotlib.pyplot as plt


def invoke_model(actor_agent, obs, exp, ep, time, agent_id):
    # parse observation
    job_dags, enb_adj, num_avail_position, \
     frontier_nodes, action_map, servers, curr_time = obs
    a = job_dags.to_list()[:]
    b = servers[:]
    exp['job_dag'].append(a)
    exp['servers'].append(b)
    # invoking the learning model

    node_act, enb_act, \
        node_act_probs, enb_act_probs, \
        node_inputs, job_inputs, job_left_inputs, enb_inputs, \
        node_valid_mask, enb_valid_mask, \
        gcn_mats, gcn_masks, summ_mats, \
        running_dags_mat, dag_summ_backward_map, dag_enb_summ_backward_map, node_enb_sum_backward_map,\
        job_dags_changed, frontier_nodes_length = \
            actor_agent.invoke_model(obs, ep, time, agent_id)

    node = action_map[node_act[0]]

    # greedy
    # waste_time_min = 100000000
    # target_server = -1
    # if len(node.parent_nodes) != 0 and len(node.child_nodes) != 0:
    #     for j in range(args.enb_num + 1):
    #         if enb_valid_mask[0, node_act[0], j] == 1:
    #             if j != args.enb_num:
    #                 est_by_server = max(servers[j].avail_time, curr_time)
    #                 est_by_parents = get_node_est_by_parents(node, j, enb_adj, curr_time)
    #             else:
    #                 est_by_server = max(servers[node.job_dag.source_id].avail_time, curr_time)
    #                 est_by_parents = get_node_est_by_parents(node, node.job_dag.source_id, enb_adj, curr_time)
    #             waste_time = abs(est_by_server - est_by_parents)
    #             if waste_time < waste_time_min:
    #                 waste_time_min = waste_time
    #                 target_server = j
    #     assert target_server != -1
    #     assert waste_time_min >= 0
    #     enb_act[0, node_act[0]] = target_server

    # min = 999999999999
    # target_server = -1
    # if len(node.parent_nodes) != 0 and len(node.child_nodes) != 0:
    #     for j in range(args.enb_num + 1):
    #         if enb_valid_mask[0, node_act[0], j] == 1:
    #             if j < args.enb_num:
    #                 est = max(servers[j].avail_time, get_node_est_by_parents(node, j, enb_adj, curr_time),
    #                           curr_time) + node.workload / servers[j].computing_power
    #             elif j == args.enb_num:
    #                 est = max(servers[node.job_dag.source_id].avail_time,
    #                           get_node_est_by_parents(node, node.job_dag.source_id, enb_adj, curr_time),
    #                           curr_time) + node.workload / servers[node.job_dag.source_id].computing_power
    #             if est < min:
    #                 min = est
    #                 target_server = j
    #     assert target_server != -1
    #     enb_act[0, node_act[0]] = target_server

    assert node_valid_mask[0, node_act[0]] == 1
    assert enb_valid_mask[0, node_act[0], enb_act[0, node_act[0]]] == 1

    if len(node.child_nodes) == 0 or len(node.parent_nodes) == 0:
        assert enb_valid_mask[0, node_act[0], args.enb_num] == 1
        assert enb_act[0, node_act[0]] == args.enb_num
        for server_idx in range(args.enb_num):
            assert enb_valid_mask[0, node_act[0], server_idx] == 0

    # for storing the action vector in experience
    node_act_vec = np.zeros(node_act_probs.shape)
    node_act_vec[0, node_act[0]] = 1

    # get_tran_loss(node, get_node_est_by_parents(node, ))

    enb_act_vec = np.zeros(enb_act_probs.shape)
    enb_act_vec[0, node_act[0], enb_act[0, node_act[0]]] = 1

    # store experience
    exp["node_act"].append(node_act[0])
    exp["enb_act"].append(enb_act[0, node_act[0]])
    exp['node_inputs'].append(node_inputs)
    exp['enb_inputs'].append(enb_inputs)
    exp['job_inputs'].append(job_inputs)
    exp['job_left_inputs'].append(job_left_inputs)
    exp['summ_mats'].append(summ_mats)
    exp['running_dag_mat'].append(running_dags_mat)
    exp['node_act_vec'].append(node_act_vec)
    exp['enb_act_vec'].append(enb_act_vec)
    exp['node_valid_mask'].append(node_valid_mask)
    exp['enb_valid_mask'].append(enb_valid_mask)
    exp['job_state_change'].append(job_dags_changed)
    exp['frontier_nodes_length'].append(frontier_nodes_length)

    if job_dags_changed:
        exp['gcn_mats'].append(gcn_mats)
        exp['gcn_masks'].append(gcn_masks)
        exp['dag_summ_back_mat'].append(dag_summ_backward_map)
        exp['dag_enb_summ_backward_map'].append(dag_enb_summ_backward_map)
        exp['node_enb_sum_backward_map'].append(node_enb_sum_backward_map)

    return node, enb_act[0, node_act[0]], node_act[0]


def train_agent(agent_id, param_queue, reward_queue, adv_queue, gradient_queue):
    # model evaluation seed

    tf.set_random_seed(agent_id)

    # set up environment
    env = Environment()

    # gpu configuration
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True))
            # per_process_gpu_memory_fraction=args.worker_gpu_fraction))

    sess = tf.Session(config=config)

    # set up actor agent
    actor_agent = ActorAgent(
        sess, args.node_input_dim, args.job_input_dim,
        args.hid_dims, args.output_dim, args.max_depth)

    # collect experiences
    while True:
        # get parameters from master
        (actor_params, seed, entropy_weight, ep) = \
            param_queue.get()
        # synchronize model
        actor_agent.set_params(actor_params)

        # reset environment
        env.seed(seed)
        env.reset()

        # set up storage for experience
        # exp是一个字典，存储所有的状态，动作，即刻奖励，输入到所有神经网络所需要的转换矩阵等。
        exp = {'node_inputs': [], 'enb_inputs': [], 'job_left_inputs':[], \
               'gcn_mats': [], 'gcn_masks': [], 'job_dag':[], 'servers': [], \
               'summ_mats': [], 'running_dag_mat': [], \
               'dag_summ_back_mat': [], \
               'node_act_vec': [], 'enb_act_vec': [], \
               'node_valid_mask': [], 'enb_valid_mask': [], \
               'reward': [], 'wall_time': [],
               'job_state_change': [], 'discount': [], 'tran_loss': [], 'waste_loss':[], 'exec_loss': [], \
               'job_inputs': [], 'dag_enb_summ_backward_map':[], 'node_enb_sum_backward_map':[], "frontier_nodes_length":[], "node_act":[], "enb_act": []}

        time = 0
        try:
        # The masking functions (node_valid_mask and
        # job_valid_mask in actor_agent.py) has some
        # small chance (once in every few thousand
        # iterations) to leave some non-zero probability
        # mass for a masked-out action. This will
        # trigger the check in "node_act and job_act
        # should be valid" in actor_agent.py
        # Whenever this is detected, we throw out the
        # rollout of that iteration and try again.

            # run experiment
            obs = env.observe()
            done = False

            # initial time
            exp['wall_time'].append(env.wall_time.curr_time)

            while not done:

                node, server_id, node_idx = invoke_model(actor_agent, obs, exp, ep, time, agent_id)

                #在step中会进一步判断执行完当前动作后是否满足继续触发调度动作的条件，
                # 如果满足则继续进行循环，如果不满足触发调度动作的条件则环境会去抓取离当前时刻最近的事件（包括两种事件：）
                obs, reward, done = env.step(node, server_id, node_idx, exp)
                time += 1
                if node is not None:
                    exp['reward'].append(reward)
                    exp['wall_time'].append(env.wall_time.curr_time)

            assert len(exp['node_inputs']) == len(exp['reward'])

            # 节点的传输损失，服务器贪心阶段无用
            sum_loss = []
            for dag in env.finished_job_dags:
                for node in dag.nodes:
                    if len(node.parent_nodes) != 0 and len(node.child_nodes) != 0:
                        sum_loss.append(node.tran_loss)

            tt = env.finished_job_dags.to_list()
            mean = np.mean([j.completion_time - j.start_time \
                            for j in env.finished_job_dags])
            # get advantage term from master

            diff_time = np.array(exp['wall_time'][1:]) - \
                        np.array(exp['wall_time'][:-1])
            sum_time = 0
            sum_reward = 0
            for i in range(len(diff_time)):
                if diff_time[i] != 0:
                    sum_time += diff_time[i]
                    sum_reward += exp["reward"][i]

            avg_per_reward = sum_reward / sum_time

            # if agent_id == 0:
            #     print(1)

            reward_queue.put(
                [exp['reward'], exp['wall_time'],
                len(env.finished_job_dags), np.mean(sum_loss), # env.finished_job_dags,
                np.mean([j.completion_time - j.start_time \
                         for j in env.finished_job_dags]), sum_reward])

            (batch_adv, avg_node_loss) = adv_queue.get()
            # batch_adv, avg_node_loss = result

            # if agent_id == args.num_agents - 1:
            #     print(1)
            if batch_adv is None or avg_node_loss == 0:
                # some other agents panic for the try and the
                # main thread throw out the rollout, reset and
                # try again now
                continue

            # compute gradients
            actor_gradient, loss = compute_actor_gradients(
                actor_agent, exp, batch_adv, entropy_weight, avg_node_loss)

            tran_loss = 0
            waste_loss = 0
            exec_loss = 0
            for dag in env.finished_job_dags:
                for node in dag.nodes:
                    if len(node.parent_nodes) != 0 and len(node.child_nodes) != 0:
                        tran_loss += node.tran_loss
                        waste_loss += node.waste_loss
                        exec_loss += node.exec_loss

            # report gradient to master
            gradient_queue.put([actor_gradient, loss, tran_loss, waste_loss, exec_loss])

        except AssertionError:
            # ask the main to abort this rollout and
            # try again
            reward_queue.put(None)
            # print("AssertionError!")
            # need to still get from adv_queue to
            # prevent blocking
            (batch_adv, avg_node_loss) = adv_queue.get()


def main():
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # create result and model folder
    create_folder_if_not_exists(args.result_folder)
    create_folder_if_not_exists(args.model_folder)

    # initialize communication queues
    # num_agents表示epoches数，用来计算平局奖励作为baseline;一次迭代包括num_agents个epoches,一次迭代更新一次policy network的参数
    # params_queues存放GCN和policy network中权重参数
    params_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    # reward_queues存放每个epoch中每个动作的即刻奖励，动作的个数等于所有workflows中的节点个数之和
    reward_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    # adv_queues存放每个动作的优势函数，即当前动作的即刻奖励+衰减因子*下一个动作的即刻奖励+。。。依此类推
    adv_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    # gradient_queues存放梯度的队列，梯度即为sigcom19中公式（3）中alpha后面的一整块
    gradient_queues = [mp.Queue(1) for _ in range(args.num_agents)]

    # set up training agents
    # agents里面存放了num_agents个线程，每个线程跑一个epoch采样
    agents = []
    for i in range(args.num_agents):
        # target表示线程所要运行的函数；args表示运行函数所需要的参数
        # TODO train_agent未看
        agents.append(mp.Process(target=train_agent, args=(
            i, params_queues[i], reward_queues[i],
            adv_queues[i], gradient_queues[i])))

    # start training agents
    for i in range(args.num_agents):
        agents[i].start()

    # gpu configuration 如果有GPU允许在GPU上运行程序，否则默认在CPU上运行
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True))
            # per_process_gpu_memory_fraction=args.master_gpu_fraction))

    sess = tf.Session(config=config)

    # print("Num GPUs Available: ", len(sess.list_devices('GPU')))


    # set up actor agent
    # 通过类ActorAgent实例化一个actor_agent对象，该对象的主要功能是定义GCN（包括节点，DAG和Global三个层面的embedding的f函数和g函数，一共6个函数是不同的，所以有6个网络）和两个policy network网络结构和初始化网络参数
    # TODO ActorAgent未看
    actor_agent = ActorAgent(
        sess, args.node_input_dim, args.job_input_dim,
        args.hid_dims, args.output_dim, args.max_depth)

    # tensorboard logging
    # sum_imm_reward表示一个epoch中所有动作即刻奖励之和；sum_reward表示一个epoch中所有动作的优势函数之和减去（8个epoches上当前动作的优势函数的均值）
    # job_duration_real表示没有噪声的线程所对应的所有工作流的完成时间
    tf_logger = TFLogger(sess, [
        'actor_loss', 'entropy', 'value_loss', 'tran_loss', 'merged_loss', 'waste_loss', 'exec_loss', 'env_tran_loss', 'env_waste_loss', 'env_exec_loss', 'episode_length',
        'average_reward_per_second', 'sum_imm_reward', 'sum_reward',
        'num_jobs', 'average_job_duration', 'job_duration_real', 'tran_loss_real', 'waste_loss_real',
        'exec_loss_real', 'env_tran_loss_real', 'env_waste_loss_real', 'env_exec_loss_real',  'entropy_weight'])

    # store average reward for computing differential rewards
    # TODO 该代码没有被用到
    avg_reward_calculator = AveragePerStepReward(
        args.average_reward_storage_size)

    # initialize entropy parameters
    # TODO 该代码没有被用到
    entropy_weight = args.entropy_weight_init
    lr = args.lr

    # ---- start training process ----
    # num_ep表示迭代次数，8个线程各跑一次之后一次迭代结束，然后继续下一个迭代
    for ep in range(1, args.num_ep):
        print('training epoch', ep)

        # synchronize the model parameters for each training agent
        # actor_agent.get_params()取出包括GCN和两个policy network对应的参数
        actor_params = actor_agent.get_params()

        # send out parameters to training agents
        for i in range(args.num_agents):
            params_queues[i].put([
                actor_params, args.seed + ep, entropy_weight, ep])
        # args.seed + 1
        # storage for advantage computation
        all_rewards, all_diff_times, all_times, \
        all_num_finished_jobs, all_avg_job_duration, \
        all_node_loss, all_imm_sum_reward = [], [], [], [], [], [], []

        #8个子线程开始的时刻
        t1 = time.time()

        # get reward from agents
        # any_agent_panic用来标识当前代码是否出现逻辑错误，即触发调度动作之后但是没有可用的资源或没有可被调度的任务节点
        any_agent_panic = False

        sum_tran_loss = 0
        for i in range(args.num_agents - 1):
            #7个子线程（有噪声的）中只要有一个没有执行完则主线程挂在这里不会往下执行，即只有当所有7个线程执行完毕程序才能往下执行，
            result = reward_queues[i].get()

            if result is None:
                any_agent_panic = True
                continue
            else:
            # batch_reward表示一个epoch中每个动作的即刻奖励，如果有10个动作则是1*10的数组；batch_time表示每个动作的执行时刻，如果有10个动作则是1*10的数组；
            # num_finished_jobs表示所有已完成的工作流个数；avg_node_loss_by_one表示每个epoch中所有选取任务节点动作所对应损失的平均值，如果有10个动作则是10个动作的平均损失；
                batch_reward, batch_time, \
                    num_finished_jobs, avg_node_loss_by_one, avg_job_duration, imm_sum_reward = result

            # for dag in finised_jobs:
            #     for node in dag.nodes:
            #         if len(node.parent_nodes) != 0 and len(node.child_nodes) != 0:
            #             sum_tran_loss += node.tran_loss

            diff_time = np.array(batch_time[1:]) - \
                        np.array(batch_time[:-1])

            all_rewards.append(batch_reward)
            all_imm_sum_reward.append(imm_sum_reward)
            all_diff_times.append(diff_time)
            all_times.append(batch_time[1:])
            all_num_finished_jobs.append(num_finished_jobs)
            all_avg_job_duration.append(avg_job_duration)
            all_node_loss.append(avg_node_loss_by_one)

            # 计算平均每秒的奖励，但在我们的实验中并没有用到该变量。
            avg_reward_calculator.add_list_filter_zero(
                batch_reward, diff_time)
        # 在训练中并没有用到这个不加噪声的epoch中的奖励，仅是HZX用于观察奖励的变化趋势
        result = reward_queues[args.num_agents - 1].get()


        # 如果8个子线程中没有出错则result不为null,则将result中的值取出来
        if result is not None:
            batch_reward, batch_time, \
            num_finished_jobs, avg_node_loss_by_one, avg_job_duration, imm_sum_reward = result

            diff_time = np.array(batch_time[1:]) - \
                        np.array(batch_time[:-1])

            all_rewards.append(batch_reward)
            all_diff_times.append(diff_time)
            all_times.append(batch_time[1:])
            all_num_finished_jobs.append(num_finished_jobs)
            all_imm_sum_reward.append(imm_sum_reward)
            # all_avg_job_duration.append(avg_job_duration)

        # TODO 8个epoch中选取节点的平均损失的平均值，即将8*1的数组中8个损失值再平均得到一个平均值
        avg_node_loss = np.mean(all_node_loss)

        # 8个线程全部执行完（即每个线程中的DAG系列从初始化到全部执行完成，并不是指将DAG系列中的所有节点调度完，因为只有执行完成才能获取所有动作的奖励，并将其放到队列中）的时刻
        t2 = time.time()
        print('got reward from workers', t2 - t1, 'seconds')

        if any_agent_panic:
            # The try condition breaks in some agent (should
            # happen rarely), throw out this rollout and try
            # again for next iteration (TODO: log this event)
            print("agent_panic")
            for i in range(args.num_agents):
                adv_queues[i].put([None, 0])
            continue

        # compute differential reward
        # all_cum_reward用来存储8个epoch中每个动作的优势函数，即为8*动作数
        all_cum_reward = []
        # 计算平均每秒的奖励，在我们的工作中并没有用到该变量
        avg_per_step_reward = avg_reward_calculator.get_avg_per_step_reward()
        for i in range(args.num_agents):
            # 判断是否要启用平均每秒的奖励，由于在我们的工作中并没有用到该变量所以程序不会执行该分支
            if args.diff_reward_enabled:
                # differential reward mode on
                rewards = np.array([r - avg_per_step_reward * t for \
                    (r, t) in zip(all_rewards[i], all_diff_times[i])])
            else:
                # regular reward
                rewards = np.array([r for \
                    (r, t) in zip(all_rewards[i], all_diff_times[i])])

            cum_reward = discount(rewards, args.gamma)

            all_cum_reward.append(cum_reward)

        # compute baseline
        # 计算8个epoch中的每个epoch中的每个动作的基线值
        baselines = get_piecewise_linear_fit_baseline(all_cum_reward, all_times)

        # all_batch_adv存放8个epoch中的每个epoch中的每个动作对应公式3中大括号的值
        all_batch_adv = []

        # give worker back the advantage
        for i in range(args.num_agents):
            # batch_adv形状为1*动作数
            batch_adv = all_cum_reward[i] - baselines[i]
            all_batch_adv.append(batch_adv)
            # np.reshape(batch_adv, [len(batch_adv), 1])将batch_adv形状整型为（动作数*1）
            batch_adv = np.reshape(batch_adv, [len(batch_adv), 1])
            # batch_adv = np.reshape(all_cum_reward[i], [len(batch_adv), 1])
            # 将batch_adv和avg_node_loss的值放到adv_queues（为子线程的队列），便于子线程计算梯度。
            adv_queues[i].put([batch_adv, avg_node_loss])

        # 所有epoch中所有动作的优势函数以及baseline计算完成时刻。
        t3 = time.time()
        print('advantage ready', t3 - t2, 'seconds')

        # 所有子线程拿到动作的优势函数以及baseline之后计算每个动作的梯度，并将其存储到actor_gradients中，它的形状为8*6个GCN网络和2个policynetwork网络的总参数个数
        actor_gradients = []
        # 下面的这些变量在实际训练中没有作用，仅用于在tensorbord中观察这些变量的值
        all_action_loss = []  # for tensorboard
        all_entropy = []  # for tensorboard
        all_value_loss = []  # for tensorboard
        all_tran_loss = []
        all_waste_loss = []
        all_exec_loss = []
        all_env_tran_loss = []
        all_env_waste_loss = []
        all_env_exec_loss = []

        for i in range(args.num_agents - 1):
            (actor_gradient, loss, tran_loss, waste_loss, exec_loss) = gradient_queues[i].get()

            actor_gradients.append(actor_gradient)  # 一个episode的梯度
            all_action_loss.append(loss[0])  # adv_loss
            all_entropy.append(-loss[1] / \
                float(all_cum_reward[i].shape[0]))
            all_value_loss.append(loss[2])
            all_tran_loss.append(loss[3])
            all_waste_loss.append(loss[4])
            all_exec_loss.append(loss[5])
            all_env_tran_loss.append(tran_loss)
            all_env_waste_loss.append(waste_loss)
            all_env_exec_loss.append(exec_loss)

        (actor_gradient, loss, tran_loss, waste_loss, exec_loss) = gradient_queues[args.num_agents - 1].get()

        t4 = time.time()
        print('worker send back gradients', t4 - t3, 'seconds')


        print("平均时间：", np.mean(all_avg_job_duration))
        print("目前最佳时间：", avg_job_duration)


        temp = aggregate_gradients(actor_gradients)

        # if ep > 1000 and lr > 0.00001:
        #     lr = lr * 0.99
        actor_agent.apply_gradients(temp, args.lr)

        t5 = time.time()
        print('apply gradient', t5 - t4, 'seconds')

        # env = Environment()
        # all_avg_time = {}
        # test_agent = {}
        # # test_agents = ['random' , 'SJF', 'avg', 'FIFO']
        # test_agents = ['FIFO']
        # for scheme in test_agents:
        #     if scheme == 'random':
        #         test_agent[scheme] = DynamicPartitionAgent(3)
        #         # test_agent[scheme] = DynamicPartitionAgent(4) # greedy_random，选服务器部分是贪心
        #     elif scheme == 'SJF':
        #         test_agent[scheme] = DynamicPartitionAgent(0)
        #     elif scheme == 'avg':
        #         test_agent[scheme] = DynamicPartitionAgent(1)
        #     elif scheme == 'FIFO':
        #         test_agent[scheme] = DynamicPartitionAgent(2)
        # for scheme in test_agents:
        #     env.seed(args.seed + ep)
        #     env.reset()
        #     obs = env.observe()
        #     agent = test_agent[scheme]
        #     done = False
        #     while not done:
        #         eexp = {}
        #         eexp['tran_loss'] = []
        #         eexp['waste_loss'] = []
        #         eexp['exec_loss'] = []
        #         node, use_exec, node_length, total_num_nodes = agent.get_action(obs, scheme)
        #         obs, reward, done = env.step(node, use_exec, 0, eexp)
        #     assert len(env.finished_job_dags) == args.num_init_dags + args.num_stream_dags
        #     all_avg_time[scheme] = np.mean([j.completion_time - j.start_time \
        #                                          for j in env.finished_job_dags])
        #     tf_logger.log_test(ep, scheme, [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,all_avg_time[scheme],0,0,0,0,0,0,0,0])


        tf_logger.log(ep, [
            np.mean(all_action_loss),
            np.mean(all_entropy),
            np.mean(all_value_loss),
            np.mean(all_tran_loss),
            np.mean(all_tran_loss) + np.mean(all_action_loss),
            np.mean(all_waste_loss),
            np.mean(all_exec_loss),
            np.mean(all_env_tran_loss),
            np.mean(all_env_waste_loss),
            np.mean(all_env_exec_loss),
            np.mean([len(b) for b in baselines]),
            avg_per_step_reward * args.reward_scale,
            np.mean(all_imm_sum_reward),
            np.mean([cr[0] for cr in all_cum_reward]),
            np.mean(all_num_finished_jobs),
            np.mean(all_avg_job_duration),
            avg_job_duration,
            loss[3],
            loss[4],
            loss[5],
            tran_loss,
            waste_loss,
            exec_loss,
            entropy_weight])

        t6 = time.time()
        print('save data in tensorboard', t6 - t5, 'seconds')

        # decrease entropy weight
        entropy_weight = decrease_var(entropy_weight,
            args.entropy_weight_min, args.entropy_weight_decay)

        if ep % args.model_save_interval == 0:
            actor_agent.save_model(args.model_folder + \
                'model_ep_' + str(ep))

    sess.close()


if __name__ == '__main__':
    main()