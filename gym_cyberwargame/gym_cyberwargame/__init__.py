from gym.envs.registration import register

register(
    id='gym_cyberwargame/CyberWarGame-v0',
    entry_point='gym_cyberwargame.envs:CyberWarGameEnv',
    max_episode_steps=300,
)

register(
    id='gym_cyberwargame/CyberWarGamePettingZoo-v0',
    entry_point='gym_cyberwargame.envs:CyberWarGameEnvPettingZoo',
)