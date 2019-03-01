import gym
from gym import error, spaces, utils
from gym.utils import seeding

class State:
    def __init__(self, ):
        df = self.read_csv("../data/AAPL-Updated.csv")
        self.df = df
        self.index = 0

    def reset():
        self.index = 0

    def next(self): # updates the current index to look at when feeding observations in step function
        if self.index >= len(self.df) - 1:
            return None, True
        values = self.df.iloc[self.index].values
        self.index += 1
        return values, False

    def __get_curr_price():
        return self.df.ix[self.index,'Open']

    def shape():
        return self.df.shape

class (gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data, episode_len, initial_cash):
        # self.states = []
        self.state = None
        self.bound = 1000000
        self.initial_cash = initial_cash    # The number of cash the agent has at the start of each episode
        self.current_cash = None            # The number of cash the agent currently possesses at this timestep
        self.num_shares = None              # The number of shares the agent currently possesses at this timestep
        # self.current_price = None           # The current stock price at this timestep
        # self.current_volume = None          # The volume of the stock at this timestep
        self.observation_space = spaces.Box(low=0, high=self.initial_cash, shape=(1,))
        self.decision_space = spaces.Discrete(3)
        self.action_space = spaces.Box(low=0, high=1)
        self.episode_len = episode_len      # length of each episode

        '''
        # % Change in Prices, may be placed somewhere else
        self.min_change_price = None        # The current change in prices by minute
        self.act_change_price = None        # The current change in price by last action
        '''

    def step(self, action, current_shares):
        # Action: Hold, Buy, Sell
        # current_shares: the current number of shares we are dealing with (buy or sell)
        # Observation : [current_cash, num_shares, %change(last min), %change(last action), current_price, volume]
        # Return value: obs, reward, done, info
        #   - obs: Observation
        #   - reward: reward at this timestep
        #   - done: True or False, indicating whether this episode has reached the end
        #   - info: don't know, don't care

        price = self.state.__get_curr_price() # current price
        cost = current_shares * price # how much cost we are dealing with
        previous_portfolio = self.current_cash + self.num_shares * price

        if action == 0: #Hold
            pass

        elif action == 1: #Buy
            if cost<=current_cash:
                self.num_shares += current_shares
                self.current_cash -= cost


        elif action == 2: #Sell
            self.num_shares -= current_shares
            self.current_cash += cost

        state, done = self.state.next()
        new_price = price

        if not done:
            new_price = self.state.__get_curr_price()

        new_equity = new_price * self.num_shares    # value of stock
        current_portfolio = new_equity + self.current_cash
        reward = ((current_portfolio - previous_portfolio)/previous_portfolio)*100.0

        return state, reward, done, None

    def reset(self):
        self.state = State()
        self.initial_cash = 1000000
        state, done = self.state.next()
        return state


    def render(self, mode='human'):
        pass

    def close(self):
        pass
