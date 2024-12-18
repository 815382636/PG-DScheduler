from param import *
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import color_map as cm

plt.rcParams['font.sans-serif'] = ['SimHei']


def plot_embedding_2d(X, map, title=None):
    # 坐标缩放到[0,1]区间
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)

    # 降维后的坐标为（X[i, 0], X[i, 1]），在该位置画出对应的digits
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    plt.xlim(-0.1,1.1)
    plt.ylim(-0.1, 1.1)
    for i in range(X.shape[0]):
        plt.text(X[i, 1], X[i, 0], str(map[i]),
                color=cm.c[map[i]],
                fontdict={'weight': 'bold', 'size': 6})


    plt.show()
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    # plt.xlabel("x", font)  # X轴标签
    # plt.ylabel("y", font)  # Y轴标签

    if title is not None:
        plt.title(title,font)

    plt.savefig("./tsne_20.pdf", format="pdf")
    print(1)

def plot_embedding_2d_DAG(X, index, title=None):
    # 坐标缩放到[0,1]区间
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    # 降维后的坐标为（X[i, 0], X[i, 1]），在该位置画出对应的digits
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], str(index),
                color=plt.cm.Set1(i / 200),
                fontdict={'weight': 'bold', 'size': 9})

    if title is not None:
        plt.title(title)

    #plt.show()
    plt.savefig("tsne_" + str(index) + ".svg", format="svg")

def visualize_executor_usage(job_dags, file_path):
    exp_completion_time = int(np.ceil(np.max([
        j.completion_time for j in job_dags])))

    temp = job_dags.to_list()
    job_durations = \
        [job_dag.completion_time - \
        job_dag.start_time for job_dag in job_dags]

    # executor_occupation = np.zeros(exp_completion_time)
    # executor_limit = np.ones(exp_completion_time) * args.exec_cap

    num_jobs_in_system = np.zeros(exp_completion_time)

    for job_dag in job_dags:
        # for node in job_dag.nodes:
        #     for task in node.tasks:
        #         executor_occupation[
        #             int(task.start_time) : \
        #             int(task.finish_time)] += 1
        num_jobs_in_system[
            int(job_dag.start_time) : \
            int(job_dag.completion_time)] += 1
    #
    # executor_usage = \
    #     np.sum(executor_occupation) / np.sum(executor_limit)
    #
    fig = plt.figure()
    #
    # plt.subplot(2, 1, 1)
    # # plt.plot(executor_occupation)
    # # plt.fill_between(range(len(executor_occupation)), 0,
    # #                  executor_occupation)
    # plt.plot(moving_average(executor_occupation, 10000))
    #
    # plt.ylabel('Number of busy executors')
    # plt.title('Executor usage: ' + str(executor_usage) + \
    #           '\n average completion time: ' + \
    #           str(np.mean(job_durations)))

    plt.subplot(2, 1, 2)
    plt.plot(num_jobs_in_system)
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Number of jobs in the system')

    fig.savefig(file_path)
    plt.close(fig)


def visualize_dag_time(job_dags, servers, plot_total_time=None, plot_type='stage'):

    dags_makespan = 0
    all_nodes = []
    # 1. compute each DAG's finish time
    # so that we can visualize it later
    dags_finish_time = []
    dags_duration = []
    for dag in job_dags:
        dag_finish_time = 0
        for node in dag.nodes:
            if node.node_finish_time > dag_finish_time:
                dag_finish_time = node.node_finish_time
            all_nodes.append(node)
        dags_finish_time.append(dag_finish_time)
        assert dag_finish_time == dag.completion_time
        dags_duration.append(dag_finish_time - dag.start_time)

    maxx = max(dags_finish_time)
    if maxx <= 250:
        maxx = 251
    elif maxx <= 300:
        maxx = 301
    elif maxx <= 350:
        maxx = 351
    elif maxx <= 400:
        maxx = 401
    else:
        maxx = int(maxx)

    # 2. visualize them in a canvas
    if plot_total_time is None:
        canvas = np.ones([len(servers), maxx]) * args.canvas_base
        temp = np.ones([len(servers), maxx]) * args.canvas_base
    else:
        canvas = np.ones([len(servers), int(plot_total_time)]) * args.canvas_base

    base = 0
    bases = {}  # job_dag -> base

    for job_dag in job_dags:
        bases[job_dag] = base
        base += job_dag.num_nodes

    for node in all_nodes:

        start_time = int(node.node_start_time)
        finish_time = int(node.node_finish_time)
        server_id = node.enb.idx

        # if plot_type == 'stage':
        #
        #     canvas[server_id, start_time : finish_time] = \
        #         bases[task.node.job_dag] + task.node.idx

        if plot_type == 'app':
            canvas[server_id, start_time : finish_time] = \
                node.job_dag.job_idx
            temp[server_id, start_time : finish_time] = \
                job_dags.index(node.job_dag)

    return canvas, dags_finish_time, dags_duration


def visualize_dag_time_save_pdf(
        job_dags, executors, file_path, plot_total_time=None, plot_type='stage'):

    canvas, dag_finish_time, dags_duration = \
        visualize_dag_time(job_dags, executors, plot_total_time, plot_type)

    fig = plt.figure()

    total = 0;
    for finish_time in dag_finish_time:
        if finish_time > total:
            total = finish_time;
    # canvas
    plt.imshow(canvas, interpolation='nearest', aspect='auto',origin='lower')
    # plt.colorbar()
    # each dag finish time
    for finish_time in dag_finish_time:
        plt.plot([finish_time, finish_time],
                 [- 0.5, len(executors) - 0.5], 'r')
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    plt.xticks(fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')
    plt.tick_params(labelsize=10)
    font = {'weight': 'normal', 'size': 15}
    plt.xlabel("时间（秒）",font)
    plt.ylabel("执行位置",font)
    plt.title('所有工作流的总完成时间:' + str('{:.10f}'.format(total))+"\n 所有工作流的平均完成时间: " +
                str('{:.10f}'.format(np.mean(dags_duration))),font)

    ax = plt.gca()

    label = ["${C}$", "${eNB_1}$", "${eNB_2}$", "${eNB_3}$", "${D_1}$", "${D_2}$", "${D_3}$", "${D_4}$",
             "${D_5}$", "${D_6}$"]

    # label = ["temp","${eNB_1}$","${eNB_2}$","${eNB_3}$","${eNB_4}$","${D_1}$","${D_2}$","${D_3}$","${D_4}$","${D_5}$","${D_6}$"]
    plt.yticks([0,1,2,3,4,5,6,7,8,9], label)
    # ax.set_yticklabels(label)

    fig.savefig(file_path, format='svg',bbox_inches='tight')
    plt.close(fig)

