# Reinforcement Learning in Finance

## Description
CAIS++ Spring 2019 Project: Building an Agent to Trade with Reinforcement Learning

## Timeline
- February 3rd:
  - In meeting:
    - First Meeting, Environment Set-Up, DQN Explanation, Project Planning
  - Homework:
    - Read first three chapter of [Spinning Up](https://spinningup.openai.com/en/latest/)
    - Watch the first half of [Stanford RL Lecture](https://www.youtube.com/watch?v=lvoHnicueoE&t=498s)
    - Code up your own LSTM on the AAPL data. Check each other's out for inspiration, find online resources, ask questions in the slack. Should be a useful exercise for everyone!
    - (Optional: Watch [LSTM Intro Video](https://www.youtube.com/watch?v=WCUNPb-5EYI))
- February 10th: Working LSTM Model
  - State:
    - Current Stock Price
    - percent change from (n-1) to n open
  - Action Space: Buy, Sell, Hold (3-dimensional continuous) as percentage of bankroll
  - Reward:
    - Life Span (define maximum length agent can interact with environment)
      - Receives reward based on profit/loss at the end
      - Sparse reward, harder to train
  - Punishment
      - Test extra severe punishment for losses
      - set thresh-hold time before it can trade again based on punishment
  - Architecture of model
      - One day of encoding LSTM
          - Observation, no actions taken
      - Second day of Policy Net
          - Based on methodology learned from encoding
          - Actions are taken
      - Two-day batches
  - Model Dimensions
      - Encoding LSTM
          - #layers of LSTM, #layers of FCs
          - input size, hidden units size, encoding vector size
      - Policy LSTM
          - input size (state space size)
          - output size (action space size: 3d continuous)
  - Homework:
      - Jincheng and Yang: Begin building Encoding / Policy Net Models
      - Chris: Look through Andrew's current LSTM model
      - Grant: Do the preprocessing data
      - Tomas: - Continue working on RL architecture
               - Make graph of prices + volume over batch
               - Visualize price gaps  
  - Pre-Process Data
  - Visualization
  - Gym Trading Environment
  - Integrate LSTM into DDDQN
- February 18th: Working DQN
  - Done for homework
    - Built first policy gradient model (Jincheng)
    - Worked on data pre-processing (Tomas)
  - Today's plan
    - Data pre-processing
    - Use data as input into the gym
    - Finalize the model
- February 24th: Work day
  - Finish pre-processing
  - Finish trading gym
      - simulate.py
          - Change action 'quit' to quit when timeseries ends
          - change time series to remove seconds
      - series_env.py
          - in class seriesenv
            - do not need daily_start_time, daily_end_time
            - remove randomization of start index (in def 'seed')
  - Finish pipelining
<<<<<<< HEAD
- February 28th: Hackathon
  - TODO
    - Review the current reward function in series_env
    - Finish building the dataset
    - Merge dataset with environment and test
    - Begin building the model
    - Create sine wave csv for testing 

- March 3rd: Working Actor-Critic Model
- March 10th: Add details like trading costs, slippage, and ask-bid spread; compute performance statistics; data visualization


## Long-Term To Do's
=======
- March 3rd:
  - Work on implementing LSTM (Chris & Caroline)
  - Create test datasets (Grant)
  - Integrate dataset with gym (Yang & Jincheng)
- March 31th:
  - Finish LSTM
  -
### Outstanding To Do's:
- Working Actor-Critic Model
- Add details like trading costs, slippage, and ask-bid spread; compute performance statistics; data visualization
>>>>>>> 13bd296a9295db93c71581f958b043ea734bdeda
- Build back testing environment
- Integrate NLP sentiment analysis as feature
- Add more indicators to model
- Clean up README
- Do we hold positions overnight? I think initially no. There are also weird jumps over holidays and weekends.
- Take into account high, low, close, volume data

## Interesting Resources

#### Reinforcement Learning Education
- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/index.html)
- [UCL + DeepMind Lecture Series](https://www.youtube.com/playlist?list=PLqYmG7hTraZDNJre23vqCGIVpfZ_K2RZs&app=desktop)
- [Stanford CS234: Reinforcement Learning](http://web.stanford.edu/class/cs234/CS234Win2018/index.html)

#### Papers
###### Primary
- [Financial Trading as a Game: A Deep Reinforcement Learning Approach](https://arxiv.org/abs/1807.02787)
- [Practical Deep Reinforcement Learning Approach for Stock Trading](https://arxiv.org/pdf/1811.07522.pdf)

###### Secondary
- [Model-based Deep Reinforcement Learning for Dynamic Portfolio Optimization](https://arxiv.org/pdf/1901.08740.pdf)

#### Medium Articles
- [Learning to trade with RL](https://medium.com/@andytwigg/learning-to-trade-with-deep-rl-666ed6bbd921)
