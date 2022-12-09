from powergym.powergym.env_register import make_env

env = make_env('13Bus')

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
