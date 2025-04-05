from ast import arg
import os
from param import *
from utils import *
from spark_env.node import *
from spark_env.job_dag import *
from spark_env.construct import construct
import json


def load_job(file_path, query_size, query_idx, wall_time, np_random, np_random_workload, source_device, job_idx):
    query_path = file_path + query_size + '/'
    
    adj_mat = np.load(
        query_path + 'adj_mat_' + str(query_idx) + '.npy', allow_pickle=True)

    assert adj_mat.shape[0] == adj_mat.shape[1]
    assert 1 <= int(query_idx) <= 100
    assert query_size == '10g'

    num_nodes = adj_mat.shape[0]
    nodes = []
    for n in range(num_nodes):

        # generate a node
        workload = np_random_workload.randint(10, 100)
        node = Node(n, "node" + str(n), wall_time, np_random, workload)
        nodes.append(node)

    # parent and child node info
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_mat[i, j] == 1:
                nodes[i].child_nodes.append(nodes[j])
                nodes[j].parent_nodes.append(nodes[i])

    roots = []

    # initialize descendant nodes
    for node in nodes:
        if len(node.parent_nodes) == 0:  # root
            get_descendant_data(node)
            roots.append(node)

    for node in nodes:
        node.output_size = node.workload * 0.1
        if len(node.child_nodes) == 0:
            node.output_size = 0.00000001

    for node in nodes:
        for parent in node.parent_nodes:
            node.input_size += parent.output_size


    # generate DAG
    job_dag = JobDAG(nodes, adj_mat,
        args.query_type + '-' + query_size + '-' + str(query_idx), args.enb_num + source_device, job_idx)

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

def generate_alibaba_jobs(np_random, np_random_workload, timeline, wall_time):
    pass


def generate_tpch_jobs(np_random, np_random_workload, timeline, wall_time):

    job_dags = OrderedSet()
    t = 0
    jobs = []
    with open("./spark_env/jobs.json", "r") as rf:
        data = json.load(rf)
        for i in data:
            jobs.append(i)

    # 不放回采样
    tpch_group = list(range(args.tpch_num))

    tpch_idx = []
    for _ in range(args.num_init_dags + args.num_stream_dags):
        idx = (np_random.choice(tpch_group))
        tpch_group.remove(idx)
        tpch_idx.append(idx + 1)

    for idx in range(args.num_init_dags):
        query_idx = str(idx + 1)
        query_size = args.tpch_size[np_random.randint(len(args.tpch_size))]
        source_device = idx % args.mobile_device_num

        # generate job
        if args.test_size:
            job_dag = load_job(
                args.job_folder, query_size, query_idx, wall_time, np_random, np_random_workload, source_device, idx)
        else:
            job_dag = construct(jobs[idx + 1], wall_time, np_random, source_device, idx)
        # job already arrived, put in job_dags
        job_dag.start_time = t
        job_dag.arrived = True
        job_dags.add(job_dag)

    for idx in range(args.num_stream_dags):
        # poisson process
        t += int(np_random.exponential(args.stream_interval))
        # uniform distribution

        source_device = (idx + args.num_init_dags) % args.mobile_device_num
        query_size = args.tpch_size[np_random.randint(len(args.tpch_size))]
        # query_idx = str(np_random.randint(args.tpch_num) + 1)
        query_idx = str(idx + args.num_init_dags + 1)

        # generate job
        if args.test_size:
            job_dag = load_job(
                args.job_folder, query_size, query_idx, wall_time, np_random, np_random_workload, source_device, idx + args.num_init_dags)
        else:
            job_dag = construct(jobs[idx + 1 + args.num_init_dags], wall_time, np_random, source_device, idx)
        # push into timeline
        job_dag.start_time = t
        assert job_dag is not None
        timeline.push(t, job_dag)
        jj = job_dags.to_list()

    return job_dags


def generate_jobs(np_random, np_random_workload, timeline, wall_time):
    if args.query_type == 'tpch':
        job_dags = generate_tpch_jobs(np_random, np_random_workload, timeline, wall_time)

    elif args.query_type == 'alibaba':
        job_dags = generate_alibaba_jobs(np_random, np_random_workload, timeline, wall_time)

    else:
        print('Invalid query type ' + args.query_type)
        exit(1)

    return job_dags
