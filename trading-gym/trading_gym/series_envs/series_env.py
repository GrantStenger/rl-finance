from gym import Env, logger
from gym.spaces import Box, Discrete

import pandas as pd
import numpy as np
import os
import os.path as osp
import random

import util.time_helper as th


class PnlSnapshot:
    def __init__(self, ):
        self.m_net_position = 0
        self.m_avg_open_price = 0
        self.m_net_investment = 0
        self.m_realized_pnl = 0
        self.m_unrealized_pnl = 0
        self.m_total_pnl = 0

    # buy_or_sell: 1 is buy, 2 is sell
    def update_by_tradefeed(self, buy_or_sell, traded_price, traded_quantity):
        assert (buy_or_sell == 1 or buy_or_sell == 2)

        if buy_or_sell == 2 and (traded_quantity > self.m_net_position):
            # Do nothing, to stop us from shorting the stock
            return False

        if buy_or_sell == 1 and self.m_net_position >= 1:
            # Do not allow to buy twice
            return False

        if buy_or_sell == 2 and self.m_net_position <= -1:
            raise ValueError('We should not have negative net poistion')



        # buy: positive position, sell: negative position
        quantity_with_direction = traded_quantity if buy_or_sell == 1 else (-1) * traded_quantity
        is_still_open = (self.m_net_position * quantity_with_direction) >= 0

        # net investment
        self.m_net_investment = max( self.m_net_investment, abs( self.m_net_position * self.m_avg_open_price  ) )
        # realized pnl
        if not is_still_open:
            # Remember to keep the sign as the net position
            self.m_realized_pnl += ( traded_price - self.m_avg_open_price ) * \
                min(
                    abs(quantity_with_direction),
                    abs(self.m_net_position)
                ) * ( abs(self.m_net_position) / self.m_net_position )
        # total pnl
        self.m_total_pnl = self.m_realized_pnl + self.m_unrealized_pnl
        # avg open price
        if is_still_open:
            self.m_avg_open_price = ( ( self.m_avg_open_price * self.m_net_position ) +
                ( traded_price * quantity_with_direction ) ) / ( self.m_net_position + quantity_with_direction )
        else:
            # Check if it is close-and-open
            if traded_quantity > abs(self.m_net_position):
                self.m_avg_open_price = traded_price
        # net position
        self.m_net_position += quantity_with_direction

        return True

    def update_by_marketdata(self, last_price):
        self.m_unrealized_pnl = ( last_price - self.m_avg_open_price ) * self.m_net_position
        self.m_total_pnl = self.m_realized_pnl + self.m_unrealized_pnl






class SeriesEnv(Env):
    def __init__(self, base_path, symb, start_date, end_date,
            daily_start_time, daily_end_time, preproc=None):
        window_length = 32
        self.calc_pct = True

        self.observation_space = Box(low=0, high=1, shape=(window_length,5),
                dtype=np.float32)

        # have three options. Buy, sell or hold
        self.action_space = Discrete(3)

        self.window_length = window_length

        df = pd.read_csv(osp.join(base_path, symb + '.csv'),
                            parse_dates=[0],
                            infer_datetime_format=False,
                            index_col=0)

        daily_start_time = th.mk_time_hours(daily_start_time)
        daily_end_time = th.mk_time_hours(daily_end_time)

        if start_date is not None:
            df = df[df.index >= start_date]
        if end_date is not None:
            df = df[df.index <= end_date]

        day_hours = df.index.hour + df.index.minute/60
        df = df[(day_hours >= daily_start_time) & (day_hours <= daily_end_time)]

        self.actions = []

        self.data = df[['open','high', 'low', 'close', 'volume']]
        replace_data = preproc.preproc(self.data.values)
        self.data = pd.DataFrame(replace_data,
                index=self.data.index[1:],
                columns=self.data.columns)
        self.cur_i = 0
        self.rand_seed()

    def render(self, mode='human'):
        raise NotImplemented('No render function')

    def rand_seed(self):
        self.seed(random.randint(0, len(self.data)))

    """
    Seeds the environment to start at the index after s that is the first new
    day
    """
    def seed(self, s):
        self.cur_i = s % len(self.data)
        use_date = self.data.index[self.cur_i].date()

        scan_i = self.cur_i
        while scan_i >= 0:
            if use_date != self.data.index[scan_i].date():
                break

            scan_i -= 1

        self.cur_i = scan_i + 1

    def get_net_pos(self):
        return self.pnl.m_net_position

    def get_cur_date(self):
        return self.data.index[self.cur_i]

    def __get_obs(self):
        done = False

        # Finish if we have reached the end of our data.
        if (self.cur_i + self.window_length) >= len(self.data):
            self.cur_i = 0
            done = True

        # Get our observation
        obs = self.data.iloc[self.cur_i:self.cur_i + self.window_length]

        start_date = obs.index[0].date()
        end_date = obs.index[-1].date()

        # Also end if the day is over
        if start_date != end_date:
            done = True
            self.cur_i = self.cur_i + self.window_length
        elif self.cur_i != 0:
            done = False

        self.cur_i += 1

        return obs.values, obs['close'].iloc[-1], done


    def step(self, action):
        obs, last_price, done = self.__get_obs()

        reward = 0

        if not done:
            self.pnl.update_by_marketdata(last_price)

            if action != 0:
                result = self.pnl.update_by_tradefeed(action, last_price, 1)
                if action == 1:
                    a = 'Buy'
                elif action == 2:
                    a = 'Sell'

                if result:
                    self.actions.append([a, last_price, self.get_cur_date()])

        reward = self.pnl.m_total_pnl
        if done:
            self.pnl = None

        return obs, reward, done, {}


    def reset(self):
        self.actions = []
        obs = self.__get_obs()[0]
        self.cur_i -= 1
        self.pnl = PnlSnapshot()

        return obs


