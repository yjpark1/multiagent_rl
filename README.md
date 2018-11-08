# Multi-Agent Reinforcement algorithms with Particle Environment (OpenAI) using pytorch

<Implemented methods>

1) ddpg.py: bidirectional LSTM actor + LSTM critic + DDPG

2) model_ddpg.py: bidirectional LSTM actor + LSTM critic + DDPG + estimate next_state + estimate reward
<estimate next_state + estimate reward> is a combined method between model-free RL and model-based RL.  
It shows significantly improved performance in particle env. simple_spread scenarios

3) model_rdpg.py: bidirectional LSTM actor + LSTM critic + RDPG (recurrent DPG) + estimate next_state + estimate reward
 This algorithm currently shows degraded performance than others in particle env. simple_spread scenarios.
