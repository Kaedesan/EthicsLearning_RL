import argparse
import pickle
from drive_mix_v2 import DrivingMix2
import numpy as np
import pandas as pd
import time

# Different ethical state for the negative, positive and mixed policies, but same general state for learning

parser = argparse.ArgumentParser(description='ethical agent')

parser.add_argument('--p_ethical', action='store_true',
                    help='indicate whether learn the Rescuing policy')
parser.add_argument('--n_ethical', action='store_true',
                    help='indicate whether learn the Avoiding policy')
parser.add_argument('--m_ethical', action='store_true',
                    help='indicate whether learn the Mixing policy')
parser.add_argument('--c', type=float, default=0.9,
                    help='a parameter to determine the human policy (default: 0.6)')
parser.add_argument('--cn', type=float, default=2,
                    help='scale of the additioal punishment (default: 2)')
parser.add_argument('--cp', type=float, default=2,
                    help='scale of the additional reward (default: 2)')
parser.add_argument('--taun', type=float, default=0.2,
                    help='threshold to determine negatively ethical behavior (default: 0.2)')
parser.add_argument('--taup', type=float, default=0.55,
                    help='threshold to determine positvely ethical behavior (default: 0.55)')
parser.add_argument('--temp', type=float, default=0.7,
                    help='the temperature parameter for Q learning policy (default: 0.7)')
parser.add_argument('--seed', type=int, default=1234,
                    help='random seed (default: 1234)')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate (default: 0.1)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor (default: 0.99)')
parser.add_argument('--num_episodes', type=int, default=2000,
                    help='number of episdoes (default: 1000)')
parser.add_argument('--verbose', action='store_true',
                    help='show log')
parser.add_argument('--record_beg', type=int, default=600,
                    help='begin to record trajectories')
parser.add_argument('--id', type=str, default= '0',
                    help='identify the experiment')
args = parser.parse_args()

actions = range(3)

np.random.seed(args.seed)
Q = {}

if args.n_ethical == True:
    dr = DrivingMix2(ishuman_n=True)
    policy_name = 'human_n_mix'
elif args.p_ethical == True:
    dr = DrivingMix2(ishuman_p=True)
    policy_name = 'human_p_mix'
else:
    dr = DrivingMix2(ishuman_m=True)
    policy_name = 'human_m_mix'

trajectory = {}
episode_rewards = []
collisions = []
cat_hits = []
elders_saved = []

def kl_div(p1, p2):
    total = 0.
    for idx in range(len(p1)):
        total += -p1[idx]*np.log(p2[idx]/p1[idx])
    return total

for cnt in range(args.num_episodes):
    state = dr.reset()
    #state = state[:2]
    rewards = 0.
    prev_pair = None
    prev_reward = None
    frame = 0

    while True:
        frame += 1
        probs = []
        for action in actions:
            try:
                probs.append(np.e**(Q[(state, action)]/args.temp))
            except:
                Q[(state, action)] = np.random.randn()
                probs.append(np.e**(Q[(state, action)]/args.temp))

        total = sum(probs)
        probs = [p / total for p in probs]

        action = np.random.choice(3, 1, p=probs)[0]
        if args.verbose: print(probs, state, action)


        if args.n_ethical == True:
            ethical_state = (state[2], state[5], state[8], state[10])

        elif args.p_ethical == True:
            ethical_state = (state[3], state[6], state[9])

        else:
            ethical_state = (state[2], state[3], state[5], state[6], state[8], state[9], state[10])


        if cnt > args.record_beg:
            try:
                trajectory[(ethical_state, action)] += 1
            except:
                trajectory[(ethical_state, action)] = 1


        if prev_pair is not None:
            Q[prev_pair] = Q[prev_pair] + args.lr * (prev_reward + args.gamma * Q[(state, action)] - Q[prev_pair])
        next_state, reward, done = dr.step(action)

        prev_pair = (state, action)
        prev_reward = reward
        rewards += reward
        if done:
            Q[prev_pair] = Q[prev_pair] + args.lr * (prev_reward - Q[prev_pair])
            break
        state = next_state

    collision, cat_hit, elder_saved = dr.log()
    collisions.append(collision)
    cat_hits.append(cat_hit)
    elders_saved.append(elder_saved)
    episode_rewards.append(rewards)

    if cnt % 100 == 0:
        print('episode: {}, frame: {}, total reward: {}'.format(cnt, frame, rewards))



df = pd.DataFrame(np.array(episode_rewards))
df.to_csv('./record/{}_{:.2f}_{:.2f}_{}_steps.csv'.format(args.id, args.temp, args.gamma, policy_name), index=False)
dfp = pd.DataFrame(np.array(collisions))
dfp.to_csv('./record/{}_{:.2f}_{:.2f}_{}_collisions.csv'.format(args.id, args.cp, args.taup, policy_name), index=False)
dfn = pd.DataFrame(np.array(cat_hits))
dfn.to_csv('./record/{}_{:.2f}_{:.2f}_{}_cat_hits.csv'.format(args.id, args.cn, args.taun, policy_name), index=False)
dfpn = pd.DataFrame(np.array(elders_saved))
dfpn.to_csv('./record/{}_{:.2f}_{:.2f}_{}_elders_saved.csv'.format(args.id, args.cp, args.taup, policy_name), index=False)

with open('./policies/'+str(args.id)+'_hpolicy_drive_'+policy_name+'.pkl', 'wb') as f:
    pickle.dump(trajectory, f, pickle.HIGHEST_PROTOCOL)
