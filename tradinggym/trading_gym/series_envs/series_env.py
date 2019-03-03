import gym
from gym import error, spaces, utils
from gym.utils import seeding

class State:
    def __init__(self, df):
        # df = self.read_csv("../data/AAPL-Updated.csv")
        self.df = df
        self.index = 0

    def reset(self):
        self.index = 0

    def next(self): # updates the current index to look at when feeding observations in step function
        if self.index >= len(self.df) - 1:
            return None, True
        values = self.df.iloc[self.index].values
        self.index += 1
        return values, False

    def __get_curr_price(self):
        return self.df.ix[self.index, 'Open']

    def shape(self):
        return self.df.shape

class SeriesEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        pass

    def init(self, data, episode_len, initial_cash):
        self.state = State(data)

        self.initial_cash = initial_cash        # The number of cash the agent has at the start of each episode
        self.current_cash = initial_cash        # The number of cash the agent currently possesses at this timestep
        self.num_shares = 0                     # The number of shares the agent currently possesses at this timestep

        self.observation_space = spaces.Box(low=0, high=self.initial_cash, shape=(1,))
        self.decision_space = spaces.Discrete(3)
        self.action_space = spaces.Box(low=0, high=1)
        self.episode_len = episode_len      # length of each episode

        '''
        # % Change in Prices, may be placed somewhere else
        self.min_change_price = None        # The current change in prices by minute
        self.act_change_price = None        # The current change in price by last action
        '''

    def step(self, action):
        # Action: action is a 2-element tuple.
        #   action[0]: {0: Hold, 1: Buy, 2: Sell}
        #   the meaning of action[1] depends on the value of action[0]:
        #       action[0] == 0 (Hold): action[1] indicates nothing
        #       action[0] == 1 (Buy): action[1] is a float in (0, 1) indicating the percentage of realized cash
        #           (cash in bank account) the agent decides to trade
        #       action[0] == 2 (Sell): action[1] is an positive integer indicating the number of shares the agent
        #           decides to sell
        #
        # current_shares: the current number of shares we are dealing with (buy or sell)
        #
        # Observation : [current_cash, num_shares, %change(last min), %change(last action), current_price, volume]
        #
        # Return value: obs, reward, done, info
        #   - obs: Observation
        #   - reward: reward at this timestep
        #   - done: True or False, indicating whether this episode has reached the end
        #   - info: don't know, don't care

        assert len(action) == 2, "the action argument must be a 2-element tuple"
        decision = action[0]
        action_value = action[1]

        # get current stock price
        price = self.state.__get_curr_price()

        # calculate previous portfolio
        previous_portfolio = self.current_cash + self.num_shares * price

        # Buy
        if decision == 1:
            current_share = action_value * self.current_cash // price
            self.num_shares += current_share
            self.current_cash -= current_share * price
        # Sell
        elif decision == 2:
            current_share = action_value
            self.num_shares -= current_share
            self.current_cash += current_share * price

        state, done = self.state.next()
        new_price = price

        if not done:
            new_price = self.state.__get_curr_price()

        new_equity = new_price * self.num_shares    # value of stock
        current_portfolio = new_equity + self.current_cash

        # TODO: Reward schedule
        reward = ((current_portfolio - previous_portfolio)/previous_portfolio)*100.0

        return state, reward, done, None

    def reset(self):
        self.state.reset()
        state, done = self.state.next()
        return state

    def render(self, mode='human'):
        pass

    def close(self):
        pass
