import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from spark_env.env import Environment
from heuristic_agent import DynamicPartitionAgent
from actor_agent import ActorAgent
from spark_env.canvas import *
from param import *
from utils import *
os.environ['CUDA_VISIBLE_DEVICES']='0'


# create result folder
if not os.path.exists(args.result_folder):
    os.makedirs(args.result_folder)

# tensorflo seeding
tf.set_random_seed(args.seed)

# set up environment
env = Environment()

# set up agents
agents = {}

for scheme in args.test_schemes:
    if scheme == 'learn':
        sess = tf.Session()
        agents[scheme] = ActorAgent(
            sess, args.node_input_dim, args.job_input_dim,
            args.hid_dims, args.output_dim, args.max_depth)
    elif scheme == 'enb':
        agents[scheme] = DynamicPartitionAgent(5)
    elif scheme == 'md':
        agents[scheme] = DynamicPartitionAgent(6)
    elif scheme == 'random':
        agents[scheme] = DynamicPartitionAgent(3) # 4 是greedy_random,即服务器部分是贪心
    elif scheme == 'SJF':
        agents[scheme] = DynamicPartitionAgent(0)
    elif scheme == 'avg':
        agents[scheme] = DynamicPartitionAgent(1)
    elif scheme == 'FIFO':
        agents[scheme] = DynamicPartitionAgent(2)
    else:
        print('scheme ' + str(scheme) + ' not recognized')
        exit(1)

# store info for all schemes
all_total_reward = {}
all_avg_time = {}
all_nodes_length= {}
all_action_enb = {}
all_total_nodes = {}
#all_action_cores = {}
all_state = {}
all_finish_time = {}
all_total_time = {}
for scheme in args.test_schemes:
    all_total_reward[scheme] = []
    all_avg_time[scheme] = []
    all_nodes_length[scheme] = []
    all_total_nodes[scheme] = []
    all_action_enb[scheme] = []
    all_state[scheme] = []
    all_total_time[scheme] = []
    all_finish_time[scheme] = []

for exp in range(args.num_exp):
    print('Experiment ' + str(exp + 1) + ' of ' + str(args.num_exp))

    for scheme in args.test_schemes:
        # reset environment with seed
        env.seed(args.num_ep + exp)
        # env.seed(43)
        if scheme == "enb":
            env.reset(2)
        elif scheme == "md":
            env.reset(1)
        else:
            env.reset()
        # load an agent
        agent = agents[scheme]

        # start experiment
        obs = env.observe()

        total_reward = 0
        done = False
        temp_node = []
        total_nodes = []
        action_enb = []
        #action_cores = []
        state = []

        while not done:
            eexp = {}
            eexp['tran_loss'] = []
            eexp['waste_loss'] = []
            eexp['exec_loss'] = []
            node, use_exec, node_length, total_num_nodes = agent.get_action(obs, scheme)
            obs, reward, done = env.step(node, use_exec, 0, eexp)
            total_reward += reward
            temp_node.append(node_length)
            total_nodes.append(total_num_nodes)
            action_enb.append(use_exec)
            # action_cores.append(server_choose_cores)
            if len(node.parent_nodes) == 0 or len(node.child_nodes) == 0:
                state.append(0)
            else:
                state.append(1)

        all_action_enb[scheme].append(action_enb)
        #all_action_cores[scheme].append(action_cores)
        all_state[scheme].append(state)
        all_nodes_length[scheme].append(temp_node)
        all_total_nodes[scheme].append(total_nodes)
        all_total_reward[scheme].append(total_reward)
        all_total_time[scheme].append(obs[6])
        all_avg_time[scheme].append(np.mean([j.completion_time - j.start_time \
                         for j in env.finished_job_dags]))
        dag_finish_time = []
        for j in env.finished_job_dags:
            dag_finish_time.append(j.completion_time - j.start_time)
        all_finish_time[scheme].append(dag_finish_time)

        if args.canvs_visualization:
            visualize_dag_time_save_pdf(
                env.finished_job_dags, env.servers,
                args.result_folder + 'visualization_exp_' + \
                str(exp) + '_scheme_' + scheme + \
                '.svg', plot_type='app')
        else:
            visualize_executor_usage(env.finished_job_dags,
                args.result_folder + 'visualization_exp_' + \
                str(exp) + '_scheme_' + scheme + '.png')

    all_avg = {}
    for scheme in args.test_schemes:
        all_avg[scheme] = np.mean(all_avg_time[scheme])

    all_avg_total = {}
    for scheme in args.test_schemes:
        all_avg_total[scheme] = np.mean(all_total_time[scheme])

print(1)
