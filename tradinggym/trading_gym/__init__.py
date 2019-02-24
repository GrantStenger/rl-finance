from gym.envs.registration import register

register(
    id='SeriesEnv-v0',
    entry_point='trading_gym.series_envs:SeriesEnv',
)
