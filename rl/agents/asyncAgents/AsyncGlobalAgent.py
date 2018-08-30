'''
Global Agent

This class can be used to do for things below.
 a. create global network using network.py
 b. create websocket connection and keep it alive to interact with its local agents
 c. create global network and give same synchronize weights of global network to other local networks of local agents.
 e. update weights from gradients update which is received from local agents.
 f. send weights of global network to local networks to its agent

Design
 - Local Agent can request separately because each agent interacts with different environment and each episode duration is different. I think global agent has to have their own interval period to mean all gradients that they have received right after it get mean value

for this, we can set a few variables like below
num_size_grads_queue
 : which means the maximum count hainvg gradients of local agents in a queue
num_min_threshold
 : once the queue size is over this value, we will calculate mean calculation of graidents.

'''

import logging
import tornado.ioloop
import tornado.web
import tornado.websocket
from tornado.options import define, options

from tornado.queues import LifoQueue as LQ
from tornado.queues import QueueEmpty
from tornado.queues import Queue as Q

import pickle

from tornado import gen

from rl import util
import gym
import asyncio
import numpy as np

from cartpole_a3c.cartpole_global import agent, env

define("port", default=9044, help="run on the given port", type=int)

grads_q = Q(100)
weight_q = LQ(1)

clients = {}
action_size = env.action_space.n
state_size = env.observation_space.shape[0]


class Application(tornado.web.Application):
    def __init__(self):
        handlers = [(r"/", WSHandler)
                    ]
        settings = dict(debug=True)
        tornado.web.Application.__init__(self, handlers, **settings)

class WSHandler(tornado.websocket.WebSocketHandler):

    _lock = False
    _recent_weight = None

    def __init__(self, application, request, **kwargs):
        super(WSHandler, self).__init__(application, request, **kwargs)
        self.ws_connection = None
        self.close_code = None
        self.close_reason = None
        self.stream = None
        self._on_close_called = False
        self.q = asyncio.Queue(5)

    def open(self, *args):
        global clients

        self.local_agent_id = self.get_argument("local_agent_id")
        print('self.local_agent_id  : {}'.format(self.local_agent_id))
        # self.stream.set_nodelay(True)
        clients[self.local_agent_id] = self

    @gen.coroutine
    def consume_weights_of_network(self):
        """
        in here, we may manage a list having last sent weights of each agent to make it sure that we send new weights
        but, for now I won't consider about it. just develop it so simple
        :return:
        """
        logging.debug('consume_weights_of_network')
        try:
            if self._recent_weight is None:
                recent_weight = yield weight_q.get()
            else:
                recent_weight = weight_q.get_nowait()

            self._recent_weight = recent_weight
            logging.debug('new weight {}'.format(self._recent_weight))

        except QueueEmpty:
            logging.debug('no weights in weight queue')
            pass

        yield self.write_message(self._recent_weight, binary=True)

    @gen.coroutine
    def calculate_mean_gradient(self):
        logging.debug('start calculating mean gradient!!')

        if not self._lock:

            self._lock = True

            try:

                num_min_threshold = 10

                # here it is best place and moment to calculate the weights
                if grads_q.qsize() >= 10:
                    grads_actor_items, grads_critic_items = [], []
                    # while grads_q.empty():
                    cnt = 0
                    while cnt < num_min_threshold:
                        i = yield grads_q.get()
                        (a, c) = pickle.loads(i)
                        grads_actor_items.append(a)
                        grads_critic_items.append(c)
                        cnt = cnt + 1
                    mean_actor = np.mean(grads_actor_items, axis=0)
                    mean_critic = np.mean(grads_critic_items, axis=0)

                    logging.debug('----------------------------------------------------------')
                    logging.debug('calculating neural net weight using mean gradients {}'.format(mean_actor))
                    logging.debug('----------------------------------------------------------')
                    logging.debug('calculating neural net weight using mean gradients {}'.format(mean_critic))
                    logging.debug('----------------------------------------------------------')

                    # TODO: update target network using mean gradient
                    agent.actor_train_on_batch([i for i in mean_actor])
                    agent.critic_train_on_batch([i for i in mean_critic])

                    logging.debug('remained {} items in queue'.format(grads_q.qsize()))
                    weight_q.put(util.get_weight_with_serialized_data(agent.actor, agent.critic))
                else:
                    pass
            except Exception as e:
                logging.error(e)
            finally:
                self._lock = False

    @gen.coroutine
    def on_message(self, message):
        """
        when we receive some message we want some message handler..
        for this example i will just print message to console
        """
        logging.info("on_message: {}".format(message))

        if message == 'send':
            print('Server is going to send weight!! in q {}'.format(weight_q))
            yield self.consume_weights_of_network()
        else:
            yield grads_q.put(message)
            yield self.calculate_mean_gradient()

    def on_close(self):
        # clients.popitem(self)
        logging.info("A client disconnected!!")

class globalAgent():
    '''
        start server for receiving and sending information
    '''
    def __init__(self, ip=None, port=None):

        # first weight of two models must be appended to the queue
        ws = util.get_weight_with_serialized_data(agent.actor, agent.critic)
        weight_q.put(ws)

        # tornado.options.parse_command_line()
        app = Application()
        if port is None:
            app.listen(options.port)
        else:
            app.listen(port)

        tornado.ioloop.IOLoop.instance().start()

# TODO: remove?
# def process_global_agent(q, ip=None, port=None):
#     # first weight of two models must be appended to the queue
#     q.append(util.get_weight_with_serialized_data(actor, critic))
#
#     tornado.options.parse_command_line()
#     app = Application()
#     if port is None:
#         app.listen(options.port)
#     else:
#         app.listen(port)
#
#     tornado.ioloop.IOLoop.instance().start()


if __name__ =="__main__":
    globalAgent()

