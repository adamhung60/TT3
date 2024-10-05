from gymnasium.envs.registration import register

register(
     id="TT3",
     entry_point="tt3_env.envs:TT3Env",
)