# Ethical issues in Reinforcement learning (RL)

As part of my internship in Lip6, we worked on the extension of the code given by the [paper](https://arxiv.org/pdf/1712.04172.pdf). Read this paper for more details on the basic scenario, especially the driving one.

They propose a low-cost strategy for giving the capability to an agent to behave ethically using RL. We used their work for extending it to multi-tasking scenario by mixing the objective (avoiding cats and saving elder people) and by adding new objectives (avoiding ambulances).

We treated these issues by single and multi-objectives resolutions.

## Requirements
Packages used:
* numpy
* pandas

Python version: *3.5.2* or later

## Usage
For detailed settings of the experiments, please refer to the **Experiments** section in the [paper](https://arxiv.org/pdf/1712.04172.pdf).
Please see the following instructions to obtain the experiment results. The results will be saved in the **record** folder.


### Driving
There are two experiments called **Driving and Avoiding and Rescuing** and **Driving and Avoiding and Rescuing and Split**. In both cases, there are cars and cats in five lanes. But in the second one, we had ambulances in the traffic. The agent should avoid the cats and save the elder people from dangers. The second experiment add the avoidance of ambulance, that can be treated as a rule to follow or as an additional objective.

The basic agent is supposed to only treat cars avoidance and driving straight.

To see the performance without human trajectories, the basic agent
```Shell
cd ./Drive/
python sarsa.py
```
We present

For **Driving and Avoiding and Rescuing**, to see the performance with human trajectories, please make sure **0_hpolicy_drive_human_m_mix.pkl** exists. If not, generate the human trajectories by
```Shell
python hsarsa_mix.py --m_ethical --id 0 --num_episodes 4000
python sarsa_mix.py --m_ethical --id 0  --num_episodes 4000 --taun 0.20 --taup 0.50
```

Similarly, for **Driving and Avoiding and Rescuing and Split**, please check the existence of the **0_hpolicy_drive_human_m_mix_split.pkll** file and use for the rule-cased ambulance avoidance
```Shell
python hsarsa_mix_plit.py --m_ethical --id 0 --num_episodes 4000
python sarsa_mix_split.py --m_ethical --rule_ambulance --id 0  --num_episodes 4000 --taun 0.20 --taup 0.50
```
The file sarsa_mix_split_mo.py can be used the same way for resolving this scenarioby a multi-objectives Approach, but need further improvements.

You can find bash file for running several experiments and other ones for printing the performance for analysis.

## Results


### Driving

All the results are summed up in the report given with the code. Many parameters are taking in consideration and explained in it.
