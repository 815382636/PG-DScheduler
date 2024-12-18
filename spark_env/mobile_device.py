from queue import Queue
from param import *

class MobileDevice(object):

    def __init__(self, idx, computing_power):
        self.idx = idx
        self.node = None
        self.job = None
        self.avail_time = 0
        self.computing_power = computing_power
        self.occupied = False
        # self.node_queue = Queue(args.md_queue_size)
        self.node_list = []
        self.node_wait = Queue(args.md_queue_size_wait)

    def reset(self):
        self.job = None
        self.node = None
        self.avail_time = 0
        self.occupied = False
        # self.node_queue = Queue(args.md_queue_size)
        self.node_list = []
        self.node_wait = Queue(args.md_queue_size_wait)
