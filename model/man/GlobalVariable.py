import collections
import tensorflow as tf

# -*- coding: utf-8 -*-
######################################################
#
# Configurations for RL model
#
######################################################

##### Constant
# TODO
cuda_device = "0"

##### Variable
# TODO
global graph
graph = tf.get_default_graph()

######################################################
#
# Configurations for HTTP Server
#
######################################################

##### Constant
SERVER_TYPE = "single_proc"
PORT = 11111


##### Variable
#token = "" # message received from client
token_deque = collections.deque(maxlen=200)
service_flag = 0 # deprecated
action = None  # message returned by RL model
flag_restart = 0
release_action = True

