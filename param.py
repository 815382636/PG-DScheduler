#coding=utf-8
#dag个数
import argparse

parser = argparse.ArgumentParser(description='DAG_ML')

# -- Basic --
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
parser.add_argument('--eps', type=float, default=1e-6,
                    help='epsilon (default: 1e-6)')
parser.add_argument('--num_exp', type=int, default=10,
                    help='number of experiments (default: 10)')
parser.add_argument('--query_type', type=str, default='tpch',
                    help='query type (default: tpch)')
parser.add_argument('--job_folder', type=str, default='./spark_env/tpch/',
                    help='job folder path (default: ./spark_env/tpch/)')
parser.add_argument('--result_folder', type=str, default='./results/',
                    help='Result folder path (default: ./results)')
parser.add_argument('--model_folder', type=str, default='./models/',
                    help='Model folder path (default: ./models)')
# parser.add_argument('--model_folder', type=str, default='./models/cloud/tiny/standard/',
#                     help='Model folder path (default: ./models)')


# -- Environment --
parser.add_argument('--enb_queue_size_wait', type=int, default=1,
                    help='Number of total executors (default: 100)')
parser.add_argument('--md_queue_size_wait', type=int, default=1,
                    help='Number of total executors (default: 100)')
parser.add_argument('--num_init_dags', type=int, default=10,
                    help='Number of initial DAGs in system (default: 10)')
parser.add_argument('--num_stream_dags', type=int, default=15,
                    help='number of streaming DAGs (default: 100)')
parser.add_argument('--enb_num', type=int, default=4,
                    help='the number of enb')
parser.add_argument('--mobile_device_num', type=int, default=6,
                    help='the number of mobile device')
parser.add_argument('--stream_interval', type=int, default=5,
                    help='inter job arrival time in milliseconds (default: 100)')
parser.add_argument('--enb_cores_nums', type=int, default=6,
                    help='inter job arrival time in milliseconds (default: 100)')
parser.add_argument('--md_cores_nums', type=int, default=4,
                    help='inter job arrival time in milliseconds (default: 100)')


parser.add_argument('--reward_scale', type=float, default=10000.0,
                    help='scale the reward to some normal values (default: 100000.0)')
parser.add_argument('--alibaba', type=bool, default=False,
                    help='Use Alibaba dags (defaule: False)')
parser.add_argument('--diff_reward_enabled', type=int, default=0,
                    help='Enable differential reward (default: 0)')

# -- Evaluation --
# parser.add_argument('--test_schemes', type=str,
#                     default=['FIFO', 'SJF', 'random', 'enb', 'md','learn'], nargs='+',
#                     help='Schemes for testing the performance')
# parser.add_argument('--test_schemes', type=str,
#                     default=['SJF','FIFO','random','enb','md','learn'], nargs='+',
#                     help='Schemes for testing the performance')
parser.add_argument('--test_schemes', type=str,
                    default=['random'], nargs='+',
                    help='Schemes for testing the performance')
parser.add_argument('--tsne', type=int, default=0)


# -- TPC-H --
parser.add_argument('--tpch_size', type=str,
                    default=['10g'], nargs='+',
                    help='Numer of TPCH queries (default: [2g, 5g, 10g, 20g, 50g, 80g, 100g])')
parser.add_argument('--tpch_num', type=int, default=45,
                    help='Numer of TPCH queries (default: 22)')

# -- Visualization --
parser.add_argument('--canvs_visualization', type=int, default=1,
                    help='Enable canvs visualization (default: 1)')
parser.add_argument('--canvas_base', type=int, default=-2,
                    help='Canvas color scale bottom (default: -10)')

# -- Learning --
parser.add_argument('--node_input_dim', type=int, default=4,
                    help='node input dimensions to graph embedding (default: 5)')
parser.add_argument('--job_input_dim', type=int, default=0,
                    help='job input dimensions to graph embedding (default: 2)')
parser.add_argument('--job_left_inputs_dim', type=int, default=6)
parser.add_argument('--enb_input_dim', type=int, default=3,
                    help='job input dimensions to graph embedding (default: 2)')
parser.add_argument('--enb_input_node_dim', type=int, default=4,
                    help='job input dimensions to graph embedding (default: 2)')
parser.add_argument('--hid_dims', type=int, default=[16, 8], nargs='+',
                    help='hidden dimensions throughout graph embedding (default: [16, 8])')
parser.add_argument('--output_dim', type=int, default=8,
                    help='output dimensions throughout graph embedding (default: 8)')
parser.add_argument('--max_depth', type=int, default=8,
                    help='Maximum depth of root-leaf message passing (default: 8)')
parser.add_argument('--lr', type=float, default=0.0003,
                    help='learning rate (default: 0.001)')
parser.add_argument('--ba_size', type=int, default=64,
                    help='Batch size (default: 64)')
parser.add_argument('--gamma', type=float, default=1,
                    help='discount factor (default: 1)')
parser.add_argument('--entropy_weight_init', type=float, default=1,
                    help='Initial exploration entropy weight (default: 1)')
parser.add_argument('--entropy_weight_min', type=float, default=0.0001,
                    help='Final minimum entropy weight (default: 0.0001)')
parser.add_argument('--entropy_weight_decay', type=float, default=1e-3,
                    help='Entropy weight decay rate (default: 1e-3)')
parser.add_argument('--log_file_name', type=str, default='log',
                    help='log file name (default: log)')
parser.add_argument('--master_num_gpu', type=int, default=0,
                    help='Number of GPU cores used in master (default: 0)')
parser.add_argument('--worker_num_gpu', type=int, default=0,
                    help='Number of GPU cores used in worker (default: 0)')
parser.add_argument('--master_gpu_fraction', type=float, default=0.5,
                    help='Fraction of memory master uses in GPU (default: 0.5)')
parser.add_argument('--worker_gpu_fraction', type=float, default=0.1,
                    help='Fraction of memory worker uses in GPU (default: 0.1)')
parser.add_argument('--average_reward_storage_size', type=int, default=10000,
                    help='Storage size for computing average reward (default: 100000)')
parser.add_argument('--num_agents', type=int, default=8,
                    help='Number of parallel agents (default: 16)')
parser.add_argument('--num_ep', type=int, default=10000000,
                    help='Number of training epochs (default: 10000000)')
parser.add_argument('--learn_obj', type=str, default='mean',
                    help='Learning objective (default: mean)')
# parser.add_argument('--saved_model', type=str, default='./models/cloud/huge/12900/10800/8000/8000/3500/1400/model_ep_5000',
#                    help='Path to the saved tf model (default: None)')
# parser.add_argument('--saved_model', type=str, default='./models/cloud/model_ep_40000',
#                    help='Path to the saved tf model (default: None)')
parser.add_argument('--saved_model', type=str, default=None,
                    help='Path to the saved tf model (default: None)')
parser.add_argument('--model_save_interval', type=int, default=100,
                    help='Interval for saving Tensorflow model (default: 1000)')
parser.add_argument('--num_saved_models', type=int, default=1000,
                    help='Number of models to keep (default: 1000)')

# -- Spark interface --
parser.add_argument('--scheduler_type', type=str, default='dynamic_partition',
                    help='type of scheduling algorithm (default: dynamic_partition)')

args = parser.parse_args()
