import argparse
import pickle
from drive_mix import DrivingMix
import numpy as np
import pandas as pd
import time

# Instead of learning from a mixed policy, used both the negative and positive
# one in the reward shapping, and the policies obtained by hsarsa_mix.py (= Same
# ethical state for all policies)

parser = argparse.ArgumentParser(description='ethical agent')
parser.add_argument('--m_ethical', action='store_true',
                    help='indicate whether learn from negative trajectory')
parser.add_argument('--c', type=float, default=0.9,
                    help='a parameter to determine the human policy (default: 0.9')
parser.add_argument('--cn', type=float, default=1,
                    help='scale of the additioal punishment (default: 2)')
parser.add_argument('--cp', type=float, default=2,
                    help='scale of the additional reward (default: 2)')
parser.add_argument('--taun', type=float, default=0.2,
                    help='threshold to determine negatively ethical behavior (default: 0.2)')
parser.add_argument('--taup', type=float, default=0.5,
                    help='threshold to determine positvely ethical behavior (default: 0.5)')
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
parser.add_argument('--count_scale', type=float, default=20,
                    help='scale the total number of count (default: 20)')
parser.add_argument('--id', type=str, default= '0',
                    help='identify the experiment')
args = parser.parse_args()

actions = range(3)

if args.m_ethical:
    hnpolicy = {}
    hppolicy = {}
    hp_filename = './policies/'+str(args.id)+'_hpolicy_drive_human_p_mix.pkl'
    hn_filename = './policies/'+str(args.id)+'_hpolicy_drive_human_n_mix.pkl'

    with open(hp_filename, 'rb') as f:
        hp_trajectory = pickle.load(f)
    with open(hn_filename, 'rb') as f:
        hn_trajectory = pickle.load(f)

    for key in hn_trajectory:
        if key[0] not in hnpolicy:
            probs = []
            count = []
            for action in actions:
                try:
                    count.append(hn_trajectory[(key[0], action)])
                except:
                    count.append(0)
            total_cnt = sum(count)
            if total_cnt > args.count_scale:
                count = [p * args.count_scale / total_cnt for p in count]

            total_cnt = sum(count)
            probs = [args.c**count[action]*(1-args.c)**(total_cnt-count[action]) for action in actions]
            print(probs)
            total_prob = sum(probs)
            probs = [p / total_prob for p in probs]
            hnpolicy[key[0]] = probs

    for key in hp_trajectory:
        if key[0] not in hppolicy:
            probs = []
            count = []
            for action in actions:
                try:
                    count.append(hp_trajectory[(key[0], action)])
                except:
                    count.append(0)
            total_cnt = sum(count)
            if total_cnt > args.count_scale:
                count = [p * args.count_scale / total_cnt for p in count]

            total_cnt = sum(count)
            probs = [args.c**count[action]*(1-args.c)**(total_cnt-count[action]) for action in actions]
            print(probs)
            total_prob = sum(probs)
            probs = [p / total_prob for p in probs]
            hppolicy[key[0]] = probs

    #for key in hnpolicy:
        #print(key, hnpolicy[key])

np.random.seed(args.seed)
Q = {}

dr = DrivingMix()
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

        if prev_pair is not None:
            Q[prev_pair] = Q[prev_pair] + args.lr * (prev_reward + args.gamma * Q[(state, action)] - Q[prev_pair])
        next_state, reward, done = dr.step(action)

        H = 0
        if args.m_ethical:
            ethical_state = (state[2], state[3], state[5], state[6], state[8], state[9], state[10])
            if ethical_state in hnpolicy:
                hnprobs = hnpolicy[ethical_state]
                if hnprobs[action] < args.taun and hnprobs[action] < probs[action] and args.m_ethical: # Only negative part of the reward shaping?? = Forbide acts
                    H += -args.cn * kl_div(probs, hnprobs)
                    #print("Neg reward shapping = ", H, " at time step ", dr.timestamp)
            if ethical_state in hppolicy:
                hpprobs = hppolicy[ethical_state]
                if hpprobs[action] > args.taup and hpprobs[action] > probs[action] and args.m_ethical: # Only positive part of the reward shaping?? = Push to act
                    H += args.cp * kl_div(probs, hpprobs)
                    #print("Pos reward shapping = ", H, " at time step ", dr.timestamp)
            reward += H

        prev_pair = (state, action)
        prev_reward = reward
        rewards += (reward - H)
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


if args.m_ethical:
    label = 'mix_ethical'
else:
    label = 'normal'

df = pd.DataFrame(np.array(episode_rewards))
df.to_csv('./record/{}_{:.2f}_{:.2f}_{}_steps.csv'.format(args.id, args.temp, args.gamma, label), index=False)
dfp = pd.DataFrame(np.array(collisions))
dfp.to_csv('./record/{}_{:.2f}_{:.2f}_{}_collisions.csv'.format(args.id, args.cp, args.taup, label), index=False)
dfn = pd.DataFrame(np.array(cat_hits))
dfn.to_csv('./record/{}_{:.2f}_{:.2f}_{}_cat_hits.csv'.format(args.id, args.cn, args.taun, label), index=False)
dfpn = pd.DataFrame(np.array(elders_saved))
dfpn.to_csv('./record/{}_{:.2f}_{:.2f}_{}_elders_saved.csv'.format(args.id, args.cp, args.taup, label), index=False)
