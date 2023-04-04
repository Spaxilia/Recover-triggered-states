# Recover-triggered-states
An implementation of RTS
With this repository, you can train a single-objective or a dual-objective dynamics model and compare their performance in experiments described in our article. The agents provided are poisoned.

## Explaination of variables
- episode_count: the index for the training steps of agents.
- env_name: the name for the gym environment  

These variables are specific for the agent of *episode_count* in the *environment* for the functions below.  
## Instructions

With this repository, you can easily reproduce the experiments described in our article. The step 0 is must be followed. Suppose you want to do experiments 1.2.3.4.5., you can just uncomment related sentences shown below.  

Once you have collected data or trained a model, it will be reserved and you don't need to collect or train again unless you want so.

0. Open Hopper, and there is the main()

1. Use the RTS to detect and defend the attack:

`before_after_defense_data(env_name, episode_count, 1)`

`plot_result(singnal=5, episode_count=int(episode_count))`  

2. Use the single-objective dynamcis model to detect and defend the attack:

`before_after_defense_data(env_name, episode_count, 0)`  

`plot_result(singnal=1, episode_count=int(episode_count))`

3. Collect data and train a RTS:

`generate_precise_date(env_name, episode_count)`  

`train_precise_dynamics(env_name, episode_count)`  

4. Collect data and train a single-objective dynamics model:

`generate_precise_date(env_name, episode_count)`  

`train_precise_dynamics(env_name, episode_count)`

5. Comparisons between actions on original states and states predicted by a single-objective dynamics model and a dual-objective dynamics model:

`new_model_path = "./Models/" + str(episode_count) + "_precise_ddpg_dm_model.pt"`

`action_difference(env_name, episode_count)`

`action_difference(env_name, episode_count, new_model_path)`

`plot_result(singnal=4, episode_count=episode_count)`

6.Change the single-step attack to consecutive attacks:

In the `before_after_defense_data()`, there is a variable `continue_attack`. Change its number to N, the attack will convert to consecutive N steps.

## Explaination of functions

### Hopper.main.py

- `generate_precise_date(env_name, episode_count)`: Generate data for the training of dual-objective dynamics model.
- `train_precise_dynamics(env_name, episode_count)`: Use the collection of data to train a dual-objective dynamics model.
- `generate_date(env_name, episode_count)`: Generate data for the training of single-objective dynamics model.
- `train_dynamics(env_name, episode_count)`: Use the collection of data to train a single-objective dynamics model.
- `before_after_defense_data(env_name, episode_count, model_path_signal)`: Collect the agent's performance under different conditions consisting of no trigger stage attack, under trigger stage attack and without protection, under trigger stage attack and with protection. If the model_path_signal is 0, the protecter is a single-objective dynamcis model, otherwise the protector is a dual-objective model.
- `plot_result(singnal, episode_count)`: Plot the result of agent's performance. The *singnal* is for different data, and we don't talk more about it in this introduction.
- `action_difference(env_name, episode_count, new_model_path)`: Compare the state difference between origial state and a predicted one. Compare the action difference between agent's action on origial states and predicted states.

