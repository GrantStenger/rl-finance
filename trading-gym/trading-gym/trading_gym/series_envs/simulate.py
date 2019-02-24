from trading_gym.series_envs.series_env import SeriesEnv

series_env = SeriesEnv()


cur_date = series_env.get_cur_date().strftime('%Y-%m-%d %H:%M:%S')
print(cur_date)

series_env.rand_seed()
cur_price = series_env.reset()[0]

print('Market simulator')
print('Actions: 1-buy/sell, 0-do nothing, q-quit')
print('-' * 20)
print('')

cur_reward = 0

while True:
    cur_date = series_env.get_cur_date().strftime('%Y-%m-%d %H:%M:%S')
    print(cur_date)
    print('')
    print('Price diff', cur_price[0])

    holding_str = ''
    print('(%i)' % (series_env.get_net_pos()))
    print('%s: %.4f' % (cur_date, cur_reward))
    action = input('Action: ')
    if action == 'q':
        break

    if 'n' in action:
        skip_steps = int(action[:-1])
        for i in range(skip_steps):
            obs, _, done, _ = series_env.step(0)
            if done:
                print('')
                print('The day is done: %.4f' % cur_reward)
                cur_reward = 0.0
    else:
        action = int(action)
        obs, reward, done, _ = series_env.step(action)
        if done:
            print('')
            print('The day is done: %.4f' % cur_reward)
            cur_reward = 0.0

        cur_reward = reward

    cur_price = obs[0]
