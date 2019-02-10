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
- February 17th: Working DQN
- February 24th: Working Actor-Critic Model
- March 3rd: Add details like trading costs, slippage, and ask-bid spread; compute performance statistics; data visualization


## Long-Term To Do's
- Build back testing environment
- Integrate NLP sentiment analysis as feature

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
