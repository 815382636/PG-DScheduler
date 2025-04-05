#coding=utf-8
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tl
import bisect
import math

from sklearn.manifold import TSNE

from param import *
from spark_env.canvas import plot_embedding_2d, plot_embedding_2d_DAG
from spark_env.enb import Enb
from spark_env.mobile_device import MobileDevice
from utils import *
from tf_op import *
from msg_passing_path import *
from gcn import GraphCNN
from gsn import GraphSNN
from spark_env.job_dag import JobDAG
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


def get_node_est_by_childs(node, enb_adj, curr_time):
    maxi = curr_time
    for child in node.child_nodes:
        if enb_adj[node.enb.idx][child.enb.idx] == 0:
            maxi = max(curr_time, child.node_finish_time)
        elif (max(curr_time, child.node_finish_time) + (child.input_size / enb_adj[node.enb.idx][child.enb.idx])) > maxi:
            maxi = max(curr_time, child.node_finish_time) + (child.input_size / enb_adj[node.enb.idx][child.enb.idx])
    return maxi    

class ActorAgent():
    def __init__(self, sess, node_input_dim, job_input_dim, hid_dims, output_dim,
                 max_depth, eps=1e-6, act_fn=leaky_relu,
                 optimizer=tf.train.AdamOptimizer, scope='actor_agent'):

        self.sess = sess
        self.node_input_dim = node_input_dim
        self.job_input_dim = job_input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.eps = eps
        self.act_fn = act_fn
        self.optimizer = optimizer
        self.scope = scope

        # for computing and storing message passing path
        self.postman = Postman()

        # node input dimension: [total_num_nodes, num_features]
        self.node_inputs = tf.placeholder(tf.float32, [None, self.node_input_dim])
        self.job_left_inputs = tf.placeholder(tf.float32, [None, args.enb_num + 2])

        self.enb_adj = tf.placeholder(tf.float32, [None, None])

        self.gcn = GraphCNN(
            self.node_inputs, self.node_input_dim, self.hid_dims,
            self.output_dim, self.max_depth, self.act_fn, self.scope)

        self.gsn = GraphSNN(
            tf.concat([self.node_inputs, self.gcn.outputs], axis=1),
            self.node_input_dim + self.output_dim, self.hid_dims,
            self.output_dim, self.act_fn, self.scope)

        # valid mask for node action ([batch_size, total_num_nodes])
        self.node_valid_mask = tf.placeholder(tf.float32, [None, None])

        self.enb_valid_mask = tf.placeholder(tf.float32, [None, None, None])

        self.enb_input = tf.placeholder(tf.float32, [None, args.enb_input_dim * (args.enb_num + 1) + args.enb_input_node_dim])
        # self.enb_input = tf.placeholder(tf.float32,[None, args.enb_input_dim * (args.enb_num + 1)])

        # map back the dag summeraization to each node ([total_num_nodes, num_dags])
        self.dag_summ_backward_map = tf.placeholder(tf.float32, [None, None])
        self.dag_enb_summ_backward_map = tf.placeholder(tf.float32, [None, None])
        self.node_enb_sum_backward_map = tf.placeholder(tf.float32, [None, None])

        # map gcn_outputs and raw_inputs to action probabilities
        # node_act_probs: [batch_size, total_num_nodes]
        # job_act_probs: [batch_size, total_num_dags]
        self.node_act_probs, self.enb_act_probs, self.temp_1, self.temp_5, self.temp_3, self.temp_4, self.temp_2 = self.actor_network(
            self.node_inputs, self.job_left_inputs, self.gcn.outputs,
            self.gsn.summaries[0], self.gsn.summaries[1],
            self.node_valid_mask,
            self.dag_summ_backward_map, self.dag_enb_summ_backward_map, self.node_enb_sum_backward_map,
            self.enb_input, self.enb_valid_mask, self.act_fn)

        logits = tf.log(self.node_act_probs)
        noise = tf.random_uniform(tf.shape(logits))
        self.node_acts = tf.argmax(logits - tf.log(-tf.log(noise)), 1)
        self.node_acts_real = tf.argmax(logits, 1)

        logits = tf.log(self.enb_act_probs)
        noise = tf.random_uniform(tf.shape(logits))
        self.enb_acts = tf.argmax(logits - tf.log(-tf.log(noise)), 2)
        self.enb_acts_real = tf.argmax(logits, 2)

        # Selected action for node, 0-1 vector ([batch_size, total_num_nodes])
        self.node_act_vec = tf.placeholder(tf.float32, [None, None])
        self.node_tran_loss = tf.placeholder(tf.float32, [None, None])
        self.node_waste_loss = tf.placeholder(tf.float32, [None, None])
        self.node_exec_loss = tf.placeholder(tf.float32, [None, None])
        # # Selected action for job, 0-1 vector ([batch_size, num_jobs, num_limits])
        self.enb_act_vec = tf.placeholder(tf.float32, [None, None, None])

        # advantage term (from Monte Calro or critic) ([batch_size, 1])
        self.adv = tf.placeholder(tf.float32, [None, 1])

        # use entropy to promote exploration, this term decays over time
        self.entropy_weight = tf.placeholder(tf.float32, ())

        # select node action probability
        self.selected_node_prob = tf.reduce_sum(tf.multiply(
            self.node_act_probs, self.node_act_vec),
            reduction_indices=1, keep_dims=True)

        # select enb action probability
        self.selected_enb_prob = tf.reduce_sum(tf.reduce_sum(tf.multiply(
            self.enb_act_probs, self.enb_act_vec),
            reduction_indices=2), reduction_indices=1, keep_dims=True)

        self.adv_loss = tf.reduce_sum(tf.multiply(
            tf.log(self.selected_node_prob * self.selected_enb_prob + \
            self.eps), -self.adv))

        # 传输损失相关，贪心选服务器阶段以下无用
        self.tran_cost = tf.reduce_sum(self.node_tran_loss, reduction_indices=1, keep_dims=True)

        self.tran_loss = tf.reduce_sum(tf.multiply(tf.log(self.selected_node_prob * self.selected_enb_prob + self.eps), -self.tran_cost))

        self.waste_loss = tf.reduce_sum(tf.multiply(tf.log(self.selected_node_prob * self.selected_enb_prob + self.eps),
                                                    -tf.reduce_sum(self.node_waste_loss, reduction_indices=1, keep_dims=True)))

        self.exec_loss = tf.reduce_sum(tf.multiply(tf.log(self.selected_node_prob * self.selected_enb_prob + self.eps),
                                                    -tf.reduce_sum(self.node_exec_loss, reduction_indices=1, keep_dims=True)))

        # node_entropy
        self.node_entropy = tf.reduce_sum(tf.multiply(
            self.node_act_probs, tf.log(self.node_act_probs + self.eps)))

        # entropy loss
        self.entropy_loss = self.node_entropy

        # normalize entropy
        self.entropy_loss /= \
            (tf.log(tf.cast(tf.shape(self.node_act_probs)[1], tf.float32)))
            # normalize over batch size (note: adv_loss is sum)
            # * tf.cast(tf.shape(self.node_act_probs)[0], tf.float32)

        # define combined loss
        self.act_loss = self.adv_loss + 0 * self.tran_loss + 0 * self.node_waste_loss + 0 * self.node_exec_loss # + \
                         # self.entropy_weight * self.entropy_loss

        # get training parameters
        self.params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

        # operations for setting network parameters
        self.input_params, self.set_params_op = \
            self.define_params_op()

        # actor gradients
        self.act_gradients = tf.gradients(self.act_loss, self.params)

        # adaptive learning rate
        self.lr_rate = tf.placeholder(tf.float32, shape=[])

        # actor optimizer
        self.act_opt = self.optimizer(self.lr_rate).minimize(self.act_loss)

        # apply gradient directly to update parameters
        self.apply_grads = self.optimizer(self.lr_rate).\
            apply_gradients(zip(self.act_gradients, self.params))

        # network paramter saver
        self.saver = tf.train.Saver(max_to_keep=args.num_saved_models)
        self.sess.run(tf.global_variables_initializer())

        if args.saved_model is not None:
            self.saver.restore(self.sess, args.saved_model)

    def actor_network(self, node_inputs, job_left_inputs,  gcn_outputs,
                      gsn_dag_summary, gsn_global_summary,
                      node_valid_mask,
                      gsn_summ_backward_map, dag_enb_summ_backward_map, node_enb_sum_backward_map,
                      enb_input, enb_valid_mask, act_fn):

        # takes output from graph embedding and raw_input from environment

        batch_size = tf.shape(node_valid_mask)[0]

        # (1) reshape node inputs to batch format
        node_inputs_reshape = tf.reshape(
            node_inputs, [batch_size, -1, self.node_input_dim])

        job_left_inputs_reshape = tf.reshape(
            job_left_inputs, [batch_size, -1, args.enb_num + 2])

        # (4) reshape gcn_outputs to batch format
        gcn_outputs_reshape = tf.reshape(
            gcn_outputs, [batch_size, -1, self.output_dim])

        # (5) reshape gsn_dag_summary to batch format
        gsn_dag_summ_reshape = tf.reshape( # 1×10×8
            gsn_dag_summary, [batch_size, -1, self.output_dim])
        gsn_summ_backward_map_extend = tf.tile(  # 1×80×10
            tf.expand_dims(gsn_summ_backward_map, axis=0), [batch_size, 1, 1])
        gsn_summ_backward_map_enb_extend = tf.tile( # 1×(80*5)×10
            tf.expand_dims(dag_enb_summ_backward_map, axis=0), [batch_size, 1, 1])
        gsn_summ_backward_map_enb_node_extend = tf.tile(  # 1×(80*5)×80
            tf.expand_dims(node_enb_sum_backward_map, axis=0), [batch_size, 1, 1])
        gsn_dag_summ_extend = tf.matmul( # 1*80*8 ## 相当于每个node拥有node.job的embedding
            gsn_summ_backward_map_extend, gsn_dag_summ_reshape)
        # gsn_dag_summ_job_extend = tf.matmul(  # 1*(80*5)*3 ## 相当于每个node的enb拥有node.job的input
        #     gsn_summ_backward_map_enb_extend, job_inputs_reshape)
        gsn_dag_summ_node_extend = tf.matmul(  # 1*(80*5)*8 ## 相当于每个node的enb拥有node的embedding
            gsn_summ_backward_map_enb_node_extend, gcn_outputs_reshape)
        gsn_dag_summ_extend_enb = tf.matmul(  # 1*(80*5)*8 ## 相当于每个node的enb都拥有node.job的embedding
            gsn_summ_backward_map_enb_extend, gsn_dag_summ_reshape)

        # (6) reshape gsn_global_summary to batch format
        gsn_global_summ_reshape = tf.reshape( # 1*1*8
            gsn_global_summary, [batch_size, -1, self.output_dim])
        gsn_global_summ_extend_job = tf.tile( # 1*10*8 ##10行一样，都拥有global的embedding
            gsn_global_summ_reshape, [1, tf.shape(gsn_dag_summ_reshape)[1], 1])
        gsn_global_summ_extend_node = tf.tile( # 1*80*8 ##80行一样，都拥有global的embedding
            gsn_global_summ_reshape, [1, tf.shape(gsn_dag_summ_extend)[1], 1])
        gsn_global_summ_extend_node_enb = tf.tile(  # 1*(80*5)*8 ##相当于每个node的enb都拥有global的embedding
            gsn_global_summ_reshape, [1, tf.shape(gsn_dag_summ_extend_enb)[1], 1])

        # (4) actor neural network
        with tf.variable_scope(self.scope):
            # -- part A, the distribution over nodes --
            merge_node = tf.concat([
                node_inputs_reshape, job_left_inputs_reshape, gcn_outputs_reshape,
                gsn_dag_summ_extend,
                gsn_global_summ_extend_node], axis=2)

            #merge_node = tf.layers.batch_normalization(merge_node)
            node_hid_0 = tl.fully_connected(merge_node, 32, activation_fn=act_fn, weights_initializer = tf.contrib.layers.variance_scaling_initializer())
            # node_hid_0 = tf.layers.batch_normalization(node_hid_0, training= False)

            node_hid_1 = tl.fully_connected(node_hid_0, 16, activation_fn=act_fn, weights_initializer = tf.contrib.layers.variance_scaling_initializer())
            # node_hid_1 = tf.layers.batch_normalization(node_hid_1, training= False)

            node_hid_2 = tl.fully_connected(node_hid_1, 8, activation_fn=act_fn, weights_initializer = tf.contrib.layers.variance_scaling_initializer())
            # node_hid_2 = tf.layers.batch_normalization(node_hid_2, training= False)

            node_outputs = tl.fully_connected(node_hid_2, 1, activation_fn=None, weights_initializer = tf.contrib.layers.variance_scaling_initializer())
            temp_1 = node_outputs
            node_outputs = tf.layers.batch_normalization(node_outputs, training= False)

            temp_3 = node_outputs

            # reshape the output dimension (batch_size, total_num_nodes)
            # TODO
            node_outputs = tf.reshape(node_outputs, [batch_size, -1]) # 1 * 80

            # node_outputs = tf.layers.batch_normalization(node_outputs, 1)

            # valid mask on node
            node_valid_mask = (node_valid_mask - 1) * 100000000000000000000000000000.0
            # maxx_abs = tf.reduce_max(math_ops.abs(node_outputs), axis=1, keepdims=True)
            # node_outputs = tf.divide(node_outputs, maxx_abs)
            node_outputs = node_outputs + node_valid_mask
            temp_4 = node_outputs

            # do masked softmax over nodes on the graph
            node_outputs = tf.nn.softmax(node_outputs, dim=-1)

            # -- part B, the distribution over enb --
            # 服务器贪心阶段以下无用

            enb_inputs_reshape = tf.reshape(enb_input, [batch_size, -1, args.enb_input_dim * (args.enb_num + 1) + args.enb_input_node_dim])
            #enb_inputs_reshape = tf.reshape(enb_input, [batch_size, -1, args.enb_input_dim * (args.enb_num + 1)])

            merge_job = tf.concat([
                enb_inputs_reshape, # 1*80*24
                gcn_outputs_reshape,# 1*80*8
                gsn_dag_summ_extend,# 1*80*8
                gsn_global_summ_extend_node], axis=2) #1*80*8

            # = enb_inputs_reshape
            merge_job = enb_inputs_reshape
            temp_2 = merge_job

            job_hid_0 = tl.fully_connected(merge_job, 32, activation_fn=act_fn)
            # job_hid_0 = tf.layers.batch_normalization(job_hid_0, training= False)

            job_hid_1 = tl.fully_connected(job_hid_0, 16, activation_fn=act_fn)
            # job_hid_1 = tf.layers.batch_normalization(job_hid_1, training= False)

            job_hid_2 = tl.fully_connected(job_hid_1, 8, activation_fn=act_fn)
            # job_hid_2 = tf.layers.batch_normalization(job_hid_2, training= False)

            enb_outputs = tl.fully_connected(job_hid_2, args.enb_num + 1, activation_fn=None)
            enb_outputs = tf.layers.batch_normalization(enb_outputs, training = False)

            enb_outputs = tf.reshape(enb_outputs, [batch_size, -1, args.enb_num + 1])
            # maxx_abs_enb = tf.reduce_max(math_ops.abs(enb_outputs), axis=2, keepdims=True) * 0.3

            # enb_outputs = tf.divide(enb_outputs, maxx_abs_enb)
            temp_5 = enb_outputs
            enb_valid_mask = (enb_valid_mask - 1) * 1000000000000000000000000000.0
            enb_outputs = enb_outputs + enb_valid_mask
            enb_outputs = tf.nn.softmax(enb_outputs, dim=-1)


            return node_outputs, enb_outputs, temp_1, temp_5, temp_3, temp_4, temp_2


    def apply_gradients(self, gradients, lr_rate):
        self.sess.run(self.apply_grads, feed_dict={
            i: d for i, d in zip(
                self.act_gradients + [self.lr_rate],
                gradients + [lr_rate])
        })

    def define_params_op(self):
        # define operations for setting network parameters
        input_params = []
        for param in self.params:
            input_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        set_params_op = []
        for idx, param in enumerate(input_params):
            set_params_op.append(self.params[idx].assign(param))
        return input_params, set_params_op

    def gcn_forward(self, node_inputs, summ_mats):
        return self.sess.run([self.gsn.summaries],
            feed_dict={i: d for i, d in zip(
                [self.node_inputs] + self.gsn.summ_mats,
                [node_inputs] + summ_mats)
        })

    def get_params(self):
        return self.sess.run(self.params)

    def save_model(self, file_path):
        self.saver.save(self.sess, file_path)

    def get_gradients(self, node_inputs, job_inputs, job_left_inputs, enb_inputs,
            node_valid_mask, enb_valid_mask,
            gcn_mats, gcn_masks, summ_mats,
            running_dags_mat, dag_summ_backward_map, dag_enb_summ_backward_map, node_enb_sum_backward_map,
            node_act_vec, node_tran_loss, node_waste_loss, node_exec_loss, enb_act_vec, adv, entropy_weight):

        act_gradients, node_act_probs, loss, enb_act_probs, selected_enb_prob, selected_node_prob, temp_1, temp_2, temp_3, temp_5, tran_cost = self.sess.run([self.act_gradients, self.node_act_probs,
            [self.adv_loss, self.entropy_loss, self.tran_loss, self.waste_loss, self.exec_loss], self.enb_act_probs, self.selected_enb_prob, self.selected_node_prob, self.temp_1,self.temp_2,self.temp_3, self.temp_5, self.tran_cost],
            feed_dict={i: d for i, d in zip(
                [self.node_inputs] + [self.enb_input] + [self.job_left_inputs] + \
                [self.node_valid_mask] + [self.enb_valid_mask] + \
                self.gcn.adj_mats + self.gcn.masks + self.gsn.summ_mats + \
                [self.dag_summ_backward_map] + [self.dag_enb_summ_backward_map] + [self.node_enb_sum_backward_map] + [self.node_act_vec] + \
                [self.node_tran_loss] + [self.node_waste_loss] + [self.node_exec_loss] + [self.enb_act_vec] + [self.adv] + [self.entropy_weight], \
                [node_inputs] + [enb_inputs] + [job_left_inputs] + \
                [node_valid_mask] + [enb_valid_mask] + \
                gcn_mats + gcn_masks + \
                [summ_mats, running_dags_mat] + \
                [dag_summ_backward_map] + [dag_enb_summ_backward_map] + [node_enb_sum_backward_map] + [node_act_vec] + \
                [node_tran_loss] + [node_waste_loss] + [node_exec_loss] + [enb_act_vec] + [adv] + [entropy_weight])
        })

        # if math.isnan(act_gradients[0][0][0]):
        #     print(1)

        return act_gradients, loss

    def predict(self, node_inputs, job_inputs, job_left_inputs, job_dags,
            node_valid_mask, enb_valid_mask, enb_adj, servers,
            gcn_mats, gcn_masks, summ_mats,
            running_dags_mat, dag_summ_backward_map, dag_enb_summ_backward_map, node_enb_sum_backward_map, action_map, curr_time, ep, time, agent_id):

        enb_input = self.get_enb_input(enb_adj, action_map, servers, curr_time)

        if agent_id != args.num_agents - 1:
            node_act_probs, enb_act_probs, embedding, temp_1, temp_2, temp_3, temp_4, temp_5, node_acts, enb_acts, temp_w = self.sess.run([self.node_act_probs, \
                self.enb_act_probs, self.gcn.outputs, self.temp_1, self.temp_2, self.temp_3, self.temp_4, self.temp_5,
                self.node_acts, self.enb_acts, self.gsn.temp_w], \
                feed_dict={i: d for i, d in zip(
                    [self.node_inputs] + [self.job_left_inputs] + \
                    [self.node_valid_mask] + [self.enb_valid_mask] + \
                    self.gcn.adj_mats + self.gcn.masks + self.gsn.summ_mats + \
                    [self.dag_summ_backward_map] + [self.dag_enb_summ_backward_map] + [self.node_enb_sum_backward_map] + \
                    [self.enb_input] + [self.enb_valid_mask], \
                    [node_inputs] + [job_left_inputs] + \
                    [node_valid_mask] + [enb_valid_mask] +  \
                    gcn_mats + gcn_masks + \
                    [summ_mats, running_dags_mat] + \
                    [dag_summ_backward_map] + [dag_enb_summ_backward_map] + [node_enb_sum_backward_map] + \
                    [enb_input] + [enb_valid_mask])
            })
        else:
            node_act_probs, enb_act_probs, embedding, temp_1, temp_2, temp_3, temp_4, temp_5, node_acts, enb_acts, temp_w = self.sess.run(
                [self.node_act_probs, self.enb_act_probs, self.gcn.outputs, self.temp_1, self.temp_2, self.temp_3, self.temp_4, self.temp_5,
                 self.node_acts_real, self.enb_acts_real, self.gsn.temp_w],
                feed_dict={i: d for i, d in zip(
                    [self.node_inputs] + [self.job_left_inputs] +\
                    [self.node_valid_mask] + [self.enb_valid_mask] + \
                    self.gcn.adj_mats + self.gcn.masks + self.gsn.summ_mats + \
                    [self.dag_summ_backward_map] + [self.dag_enb_summ_backward_map] + [self.node_enb_sum_backward_map] + \
                    [self.enb_input] + [self.enb_valid_mask], \
                    [node_inputs] + [job_left_inputs] + \
                    [node_valid_mask] + [enb_valid_mask] + \
                    gcn_mats + gcn_masks + \
                    [summ_mats, running_dags_mat] + \
                    [dag_summ_backward_map] + [dag_enb_summ_backward_map] + [node_enb_sum_backward_map] +\
                    [enb_input] + [enb_valid_mask])
                           })

        idx_map = {}
        if_tsne = 0
        # idx_map[0] = 1;idx_map[1]=24;idx_map[2]=5;idx_map[3]=2;idx_map[4]=20;idx_map[5]=0;idx_map[6]=16;idx_map[7]=22;idx_map[8]=18;idx_map[9]=23;idx_map[10]=8;idx_map[11]=7;idx_map[12]=9;idx_map[13]=3;idx_map[14]=4;idx_map[15]=12;idx_map[16]=17;idx_map[17]=14;idx_map[18]=6;idx_map[19]=13;idx_map[20]=15;idx_map[21]=18;idx_map[22]=11;idx_map[23]=21;idx_map[24]=10;
        for dag in job_dags:
            if dag.job_idx >= 19:
                if_tsne = 1
        if args.tsne == 1 and if_tsne == 1:
            tsne = TSNE(n_components=2, init='pca', random_state=0)
            X_tsne = tsne.fit_transform(embedding)

            node_idx_to_jod_idx_map = {}
            node_idx = 0
            for dag in job_dags:
                for node in dag.nodes:
                    node_idx_to_jod_idx_map[node_idx] = node.job_dag.job_idx
                    node_idx += 1

            # for dag in job_dags:
            #     num_nodes = dag.num_nodes
            #     x = tsne.fit_transform(embedding[index:index+num_nodes])
            #     index = index + num_nodes
            #     plot_embedding_2d_DAG(x[:,:], job_idx, "t-SNE 2D")
            #     job_idx += 1


            plot_embedding_2d(X_tsne[:, :], node_idx_to_jod_idx_map, "t-SNE 2D")

        if node_valid_mask[0, node_acts[0]] == 0:
            print(1)

        if enb_valid_mask[0, node_acts[0], enb_acts[0, node_acts[0]]] == 0:
            print(1)

        return node_act_probs, enb_act_probs, node_acts, enb_acts, enb_input

    def get_enb_input(self, enb_adj, action_map, servers, curr_time):
        # action_map表示当前系统中存放所有任务节点对象的列表，列表中的元素为任务节点对象
        # enb_input是一个二维数组，行为当前系统中所有任务节点的个数，列为：
        # enb_input_dim是一个1*3纬的列表，其中包括任务节点在对应服务器或MD上的执行时间，
        # 任务节点在对应服务器的最早开始时刻（因为任务节点放到不同服务器上其前驱节点的传输时间不同）减去当前时刻（即为任务在对应服务器的可用时刻减去当前时刻，
        # 即服务器要等待当前任务所有前驱任务执行完成所需等待的时间），对应服务器的可用时间（即任务节点在当前时刻可以执行，但服务器中队列还有任务在等待，
        # 即为任务节点在对应服务器或MD所需的等待时间）
        # enb_input_node_dim是1*4纬的列表，其中包括直接子节点（不包括子节点的子节点）的个数，子节点的工作负载之和，所有后代节点的个数，所有后代节点的工作负载之和
        enb_input = np.zeros([len(action_map), args.enb_input_dim * (args.enb_num + 1) + args.enb_input_node_dim])
        # enb_input = np.zeros([len(action_map), args.enb_input_dim * (args.enb_num + 1)])
        for i in range(len(action_map)):
            for j in range(args.enb_num):
                if action_map[i].is_schedulable() is True:
                    est_by_parents = (get_node_est_by_parents(action_map[i], j, enb_adj, curr_time) - curr_time) / 200
                    est_by_server = (max(servers[j].avail_time, curr_time) - curr_time) / 200
                    enb_input[i, 3 * j] = est_by_parents
                    enb_input[i, 3 * j + 1] = est_by_server
                else:
                    enb_input[i, 3 * j] = -1
                    enb_input[i, 3 * j + 1] = -1
                enb_input[i, 3 * j + 2] = action_map[i].workload / servers[j].computing_power / 15

            md_idx = action_map[i].job_dag.source_id
            if action_map[i].is_schedulable() is True:
                est_by_parents = (get_node_est_by_parents(action_map[i], md_idx, enb_adj, curr_time) - curr_time) / 200
                est_by_server = (max(servers[md_idx].avail_time, curr_time) - curr_time) / 200
                enb_input[i, 3 * args.enb_num] = est_by_parents
                enb_input[i, 3 * args.enb_num + 1] = est_by_server
                enb_input[i, 3 * args.enb_num + 2] = action_map[i].workload / servers[md_idx].computing_power / 15
            else:
                enb_input[i, 3 * args.enb_num] = -1
                enb_input[i, 3 * args.enb_num + 1] = -1
                enb_input[i, 3 * args.enb_num + 2] = action_map[i].workload / servers[md_idx].computing_power / 15

            child_workload = 0
            descent_workload = 0
            descent_num = len(action_map[i].descendant_nodes)
            for descent in action_map[i].descendant_nodes:
                descent_workload += descent.workload
            for child in action_map[i].child_nodes:
                child_workload += child.workload

            enb_input[i, args.enb_input_dim * (args.enb_num + 1)] = len(action_map[i].child_nodes) / 5
            enb_input[i, args.enb_input_dim * (args.enb_num + 1) + 1] = child_workload / 200
            enb_input[i, args.enb_input_dim * (args.enb_num + 1) + 2] = descent_num / 10
            enb_input[i, args.enb_input_dim * (args.enb_num + 1) + 3] = descent_workload / 400

        return enb_input

    def set_params(self, input_params):
        self.sess.run(self.set_params_op, feed_dict={
            i: d for i, d in zip(self.input_params, input_params)
        })

    def translate_state(self, obs, ep, time, agent_id):
        """
        Translate the observation to matrix form
        """
        job_dags, enb_adj, num_avail_position,  \
        frontier_nodes, action_map, servers, curr_time = obs

        # compute total number of nodes
        total_num_nodes = int(np.sum(job_dag.num_nodes for job_dag in job_dags))

        # job and node inputs to feed
        node_inputs = np.zeros([total_num_nodes, self.node_input_dim])
        job_inputs = np.zeros([len(job_dags), self.job_input_dim])

        # gather job level inputs
        # job_idx = 0
        # for job_dag in job_dags:
        #     job_inputs[job_idx, 0] = (curr_time - job_dag.start_time) / 100
        #     job_idx += 1

        # gather node level inputs
        node_idx = 0
        job_idx = 0
        for job_dag in job_dags:
            for node in job_dag.nodes:

                #node_inputs[node_idx, :1] = job_inputs[job_idx, :1]

                # work on the node
                node_inputs[node_idx, 0] = (node.workload - 10) / 90

                node_inputs[node_idx, 1] = node.input_size / 20
                node_inputs[node_idx, 2] = node.output_size / 20

                sum_workload = 0
                for parent in node.parent_nodes:
                    sum_workload += parent.workload
                node_inputs[node_idx, 3] = sum_workload / 200

                # if node.occupied:
                #     node_inputs[node_idx, 4] = 1
                # else:
                #     node_inputs[node_idx, 4] = -1

                node_idx += 1
            job_idx += 1

        return node_inputs, job_inputs, \
               job_dags, \
               frontier_nodes, action_map, enb_adj, servers, curr_time

    def get_valid_masks(self, job_dags, frontier_nodes,
                        action_map, servers):

        total_num_nodes = int(np.sum(
            job_dag.num_nodes for job_dag in job_dags))

        enb_valid = np.zeros([1, len(servers)])
        enb_valid_mask = np.zeros([1, total_num_nodes, args.enb_num + 1])

        for server in servers:
            if not server.occupied or not server.node_wait.full():
                enb_valid[0, server.idx] = 1

        for i in range(total_num_nodes):
            for server_idx in range(len(servers)):
                if server_idx < args.enb_num:
                     enb_valid_mask[0, i, server_idx] = enb_valid[0, server_idx]
            enb_valid_mask[0, i, args.enb_num] = enb_valid[0, action_map[i].job_dag.source_id]
            if len(action_map[i].child_nodes) == 0 or len(action_map[i].parent_nodes) == 0:
                for server_idx in range(args.enb_num):
                    enb_valid_mask[0, i, server_idx] = 0

        node_valid_mask = np.zeros([1, total_num_nodes])

        for node in frontier_nodes:
            if len(node.child_nodes) == 0 or len(node.parent_nodes) == 0:
                assert enb_valid[0, node.job_dag.source_id] == 1
            act = action_map.inverse_map[node]
            node_valid_mask[0, act] = 1

        return node_valid_mask, enb_valid_mask

    def invoke_model(self, obs, ep, time, agent_id):
        # implement this module here for training
        # (to pick up state and action to record)

        node_inputs, job_inputs, \
            job_dags, \
            frontier_nodes, \
            action_map, enb_adj, servers, curr_time = self.translate_state(obs, ep, time, agent_id)

        # get message passing path (with cache)
        gcn_mats, gcn_masks, dag_summ_backward_map, dag_enb_summ_backward_map, node_enb_sum_backward_map,\
            running_dags_mat, job_dags_changed = \
            self.postman.get_msg_path(job_dags, len(servers))

        # get node and job valid masks
        node_valid_mask, enb_valid_mask = \
            self.get_valid_masks(job_dags, frontier_nodes,
                                 action_map, servers)

        # get summarization path that ignores finished nodes
        summ_mats = get_unfinished_nodes_summ_mat(job_dags)
        # self.job_mats = get_unfinished_nodes_job_mat(job_dags)

        total_num_nodes = int(np.sum(job_dag.num_nodes for job_dag in job_dags))

        job_left_inputs = np.zeros([total_num_nodes, args.enb_num + 2])
        node_idx = 0
        for job_dag in job_dags:
            left_time = (curr_time - job_dag.start_time) / 300
            for node in job_dag.nodes:
                job_left_inputs[node_idx, 0] = left_time
                if node.is_schedulable():
                    job_left_inputs[node_idx, 1] = abs(get_node_est_by_parents(node, job_dag.source_id, enb_adj, curr_time) - max(curr_time, servers[job_dag.source_id].avail_time)) / 100
                    for i in range(args.enb_num):
                        job_left_inputs[node_idx, i+2] = abs(
                            get_node_est_by_parents(node, i, enb_adj, curr_time) - max(curr_time, servers[i].avail_time)) / 100
                    # job_left_inputs[node_idx, 2] = abs(get_node_est_by_parents(node, 0, enb_adj, curr_time) - max(curr_time, servers[0].avail_time)) / 100
                    # job_left_inputs[node_idx, 3] = abs(get_node_est_by_parents(node, 1, enb_adj, curr_time) - max(curr_time, servers[1].avail_time)) / 100
                    # job_left_inputs[node_idx, 4] = abs(get_node_est_by_parents(node, 2, enb_adj, curr_time) - max(curr_time, servers[2].avail_time)) / 100
                    # job_left_inputs[node_idx, 5] = abs(get_node_est_by_parents(node, 3, enb_adj, curr_time) - max(curr_time, servers[3].avail_time)) / 100
                    # job_left_inputs[node_idx, 6] = abs(get_node_est_by_parents(node, 4, enb_adj, curr_time) - max(curr_time, servers[4].avail_time)) / 100
                    # job_left_inputs[node_idx, 7] = abs(get_node_est_by_parents(node, 3, enb_adj, curr_time) - max(curr_time, servers[5].avail_time)) / 100
                else:
                    job_left_inputs[node_idx, 1] = -1
                    for i in range(args.enb_num):
                        job_left_inputs[node_idx, i+2] = -1
                    # job_left_inputs[node_idx, 2] = -1
                    # job_left_inputs[node_idx, 3] = -1
                    # job_left_inputs[node_idx, 4] = -1
                    # job_left_inputs[node_idx, 5] = -1
                    # job_left_inputs[node_idx, 6] = -1
                    # job_left_inputs[node_idx, 7] = -1

                node_idx += 1

        # invoke learning model
        node_act_probs, enb_act_probs, node_acts, enb_acts, enb_inputs = \
            self.predict(node_inputs, job_inputs, job_left_inputs, job_dags,
                         node_valid_mask, enb_valid_mask, enb_adj, servers, \
                         gcn_mats, gcn_masks, summ_mats, \
                         running_dags_mat, dag_summ_backward_map, dag_enb_summ_backward_map, node_enb_sum_backward_map, action_map, curr_time, ep, time, agent_id)

        # node = action_map[node_acts[0]]

        return node_acts, enb_acts, \
               node_act_probs, enb_act_probs, \
               node_inputs, job_inputs, job_left_inputs, enb_inputs, \
               node_valid_mask, enb_valid_mask, \
               gcn_mats, gcn_masks, summ_mats, \
               running_dags_mat, dag_summ_backward_map, dag_enb_summ_backward_map, node_enb_sum_backward_map, job_dags_changed, len(frontier_nodes)

    def get_action(self, obs, scheme):

        job_dags, enb_adj, num_avail_position, \
        frontier_nodes, action_map, servers, curr_time = obs

        node_act, enb_act, \
        node_act_probs, enb_act_probs, \
        node_inputs, job_inputs, job_left_inputs, enb_inputs, \
        node_valid_mask, enb_valid_mask, \
        gcn_mats, gcn_masks, summ_mats, \
        running_dags_mat, dag_summ_backward_map, dag_enb_summ_backward_map, node_enb_sum_backward_map, \
        job_dags_changed, node_length = \
            self.invoke_model(obs, 1, 1, args.num_agents - 1)

        node = action_map[node_act[0]]

        assert node_valid_mask[0, node_act[0]] == 1
        assert enb_valid_mask[0, node_act[0], enb_act[0, node_act[0]]] == 1

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

        min = 999999999999
        # target_server = -1
        # if len(node.parent_nodes) != 0 and len(node.child_nodes) != 0:
        #     for j in range(args.enb_num + 1):
        #         if enb_valid_mask[0, node_act[0], j] == 1:
        #             if j < args.enb_num:
        #                 est = max(servers[j].avail_time, get_node_est_by_parents(node, j, enb_adj, curr_time),
        #                           curr_time) + node.workload / servers[j].computing_power
        #             elif j == args.enb_num:
        #                 est = max(servers[node.job_dag.source_id].avail_time, get_node_est_by_parents(node, node.job_dag.source_id, enb_adj, curr_time),
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

        total_num_nodes = int(np.sum(job_dag.num_nodes for job_dag in job_dags))

        if scheme == 'md':
            assert enb_valid_mask[0, node_act[0], args.enb_num] == 1
            enb_act[0, node_act[0]] = args.enb_num
        elif scheme == 'enb':
            if len(node.parent_nodes) != 0 and len(node.child_nodes) != 0:
                for j in range(args.enb_num):
                    if enb_valid_mask[0, node_act[0], j] == 1:
                        est = max(servers[j].avail_time, get_node_est_by_parents(node, j, enb_adj, curr_time),
                                      curr_time) + node.workload / servers[j].computing_power
                        if est < min:
                            min = est
                            target_server = j
                assert target_server != -1
                enb_act[0, node_act[0]] = target_server

        return node, enb_act[0, node_act[0]], node_length, total_num_nodes
