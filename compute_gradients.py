from param import *
from utils import *
import math
from sparse_op import expand_sp_mat, merge_and_extend_sp_mat


def compute_actor_gradients(actor_agent, exp, batch_adv, entropy_weight, avg_tran_loss):
    batch_points = truncate_experiences(exp['job_state_change'])

    all_gradients = []
    all_loss = [[], [], 0, [], [], []]

    # episodes
    for b in range(len(batch_points) - 1):
        # need to do different batches because the
        # size of dags in state changes
        ba_start = batch_points[b]
        ba_end = batch_points[b + 1]

        # use a piece of experience
        node_inputs = np.vstack(exp['node_inputs'][ba_start : ba_end])
        job_inputs = np.vstack(exp['job_inputs'][ba_start: ba_end])
        job_left_inputs = np.vstack(exp['job_left_inputs'][ba_start: ba_end])
        enb_inputs = np.vstack(exp['enb_inputs'][ba_start : ba_end])
        node_act_vec = np.vstack(exp['node_act_vec'][ba_start : ba_end])
        node_tran_loss = np.vstack(exp['tran_loss'][ba_start: ba_end])
        node_waste_loss = np.vstack(exp['waste_loss'][ba_start: ba_end])
        node_exec_loss = np.vstack(exp['exec_loss'][ba_start: ba_end])
        enb_act_vec = np.vstack(exp['enb_act_vec'][ba_start : ba_end])
        node_valid_mask = np.vstack(exp['node_valid_mask'][ba_start : ba_end])
        enb_valid_mask = np.vstack(exp['enb_valid_mask'][ba_start : ba_end])
        summ_mats = exp['summ_mats'][ba_start : ba_end]
        running_dag_mats = exp['running_dag_mat'][ba_start : ba_end]
        adv = batch_adv[ba_start : ba_end, :]
        gcn_mats = exp['gcn_mats'][b]
        gcn_masks = exp['gcn_masks'][b]
        summ_backward_map = exp['dag_summ_back_mat'][b]
        dag_enb_summ_backward_map =  exp['dag_enb_summ_backward_map'][b]
        node_enb_sum_backward_map = exp['node_enb_sum_backward_map'][b]

        # given an episode of experience (advantage computed from baseline)
        batch_size = node_act_vec.shape[0]

        # expand sparse adj_mats
        extended_gcn_mats = expand_sp_mat(gcn_mats, batch_size)

        # extended masks
        # (on the dimension according to extended adj_mat)
        extended_gcn_masks = [np.tile(m, (batch_size, 1)) for m in gcn_masks]

        # expand sparse summ_mats
        extended_summ_mats = merge_and_extend_sp_mat(summ_mats)

        # expand sparse running_dag_mats
        extended_running_dag_mats = merge_and_extend_sp_mat(running_dag_mats)

        for i in range(node_tran_loss.shape[0]):
            for j in range(node_tran_loss.shape[1]):
                if node_tran_loss[i][j] == -1:
                    node_tran_loss[i][j] = 0
                else:
                    node_tran_loss[i][j] = avg_tran_loss - node_tran_loss[i][j]

        node_tran_loss = np.multiply(node_tran_loss, node_act_vec)

        # compute gradient
        act_gradients, loss = actor_agent.get_gradients(
            node_inputs, job_inputs, job_left_inputs, enb_inputs,
            node_valid_mask, enb_valid_mask,
            extended_gcn_mats, extended_gcn_masks,
            extended_summ_mats, extended_running_dag_mats,
            summ_backward_map, dag_enb_summ_backward_map, node_enb_sum_backward_map, node_act_vec, node_tran_loss, node_waste_loss, node_exec_loss,
            enb_act_vec, adv, entropy_weight)

        all_gradients.append(act_gradients)
        all_loss[0].append(loss[0])
        all_loss[1].append(loss[1])
        all_loss[3].append(loss[2])
        all_loss[4].append(loss[3])
        all_loss[5].append(loss[4])

    all_loss[0] = np.sum(all_loss[0])
    all_loss[1] = np.sum(all_loss[1])  # to get entropy
    all_loss[2] = np.sum(batch_adv ** 2) # time based baseline loss
    all_loss[3] = np.sum(all_loss[3])
    all_loss[4] = np.sum(all_loss[4])
    all_loss[5] = np.sum(all_loss[5])

    # aggregate all gradients from the batches
    gradients = aggregate_gradients(all_gradients)

    return gradients, all_loss
