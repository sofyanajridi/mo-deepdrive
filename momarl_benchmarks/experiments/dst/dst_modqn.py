from momarl_benchmarks.algorithms.mo_dqn import MODQN
import mo_gymnasium as mo_gym
import math

# Hyperparameters
BATCH_SIZE = 32  # 128
GAMMA = 0.99
TAU = 0.001
LR =  3e-5






from datetime import datetime

now = datetime.now()

# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y-%H:%M:%S")

wandb_name = "DST_MODQN" + dt_string

env = mo_gym.make("deep-sea-treasure-v0",float_state=True)

def utility_f(r):
    r0, r1 = r
    return 0.9*r0 + 0.1*r1

def non_lin(r):
    r0, r1 = r
    d0 = 45
    d1 = 10
    p = 10
    if r1 <= d1:
        return math.log(1 + math.exp(r0 - d0))
    else:
        return math.log(1 + math.exp(r0 - d0)) - pow((r1-d1),2) - p

# def non_lin(r):
#     r0, r1 = r
#     d0 = 45
#     d1 = 10
#     p = 10
#     if r1 <= d1:
#         return 0.9*r0 + 0.1*r1
#     else:
#         return 0.1*r0 + 0.9*r1




dqn_agent = MODQN(env, batch_size=BATCH_SIZE, gamma=GAMMA, tau=TAU, lr=LR, utility_f=non_lin)
dqn_agent.train(50_000, enable_wandb_logging="online", wandb_group_name="DST_MODQN",
                wandb_name=wandb_name)