import argparse
import pickle
from drive import Driving
import numpy as np
import pandas as pd

# Original code from the author

parser = argparse.ArgumentParser(description='ethical agent')
parser.add_argument('--ethical', action='store_true',
                    help='indicate whether learn from human trajectory or not')
parser.add_argument('--c', type=float, default=0.6,
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
parser.add_argument('--num_episodes', type=int, default=1000,
                    help='number of episdoes (default: 1000)')
parser.add_argument('--verbose', action='store_true',
                    help='show log')
parser.add_argument('--record_beg', type=int, default=600,
                    help='begin to record trajectories')
parser.add_argument('--id', type=str, default= '0',
                    help='identify the experiment')
parser.add_argument('--first_ep_rec', type=int, default= 0,
                    help='number of recorded episodes from the begining')
parser.add_argument('--last_ep_rec', type=int, default= 0,
                    help='number of recorded episodes from the end')
args = parser.parse_args()

actions = range(3)

np.random.seed(args.seed)
Q = {}

dr = Driving(ishuman_p=True)
trajectory = {}
episode_rewards = []
collisions = []
cat_hits = []

if args.first_ep_rec > 0:
    beg_record_collision = [0 for i in range(dr.sim_len+1)]
    beg_record_cat_saved = [0 for i in range(dr.sim_len+1)]

if args.last_ep_rec > 0:
    end_record_collision = [0 for i in range(dr.sim_len+1)]
    end_record_cat_saved = [0 for i in range(dr.sim_len+1)]

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
    beg_rec = False
    end_rec = False

    if (args.first_ep_rec > 0 and cnt < args.first_ep_rec):
        beg_rec = True
    elif (args.last_ep_rec > 0 and cnt > args.num_episodes - args.last_ep_rec -1):
        end_rec = True

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

        ethical_state = (state[2], state[4], state[6])
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

        if beg_rec == True:
            col, ch= dr.log()
            beg_record_collision[dr.timestamp] += col
            beg_record_cat_saved[dr.timestamp] += ch
        if end_rec == True:
            col, ch= dr.log()
            end_record_collision[dr.timestamp] += col
            end_record_cat_saved[dr.timestamp] += ch

        if done:
            Q[prev_pair] = Q[prev_pair] + args.lr * (prev_reward - Q[prev_pair])
            break
        state = next_state

    collision, cat_hit = dr.log()
    collisions.append(collision)
    cat_hits.append(cat_hit)
    episode_rewards.append(rewards)

    if cnt % 100 == 0:
        print('episode: {}, frame: {}, total reward: {}'.format(cnt, frame, rewards))

label = 'human_p'

if args.first_ep_rec > 0:
    for i in range(dr.sim_len+1):
        beg_record_collision[i] = beg_record_collision[i]/args.first_ep_rec
        beg_record_cat_saved[i] = beg_record_cat_saved[i]/args.first_ep_rec

    beg_col_ev = pd.DataFrame(np.array(beg_record_collision))
    beg_col_ev.to_csv('./record_timesteps/{}_{}_{:.2f}_{:.2f}_{}_collisions_begining.csv'.format(args.id, args.first_ep_rec, args.temp, args.gamma, label), index=False)
    beg_cats_ev = pd.DataFrame(np.array(beg_record_cat_saved))
    beg_cats_ev.to_csv('./record_timesteps/{}_{}_{:.2f}_{:.2f}_{}_cat_hits_begining.csv'.format(args.id, args.first_ep_rec, args.temp, args.gamma, label), index=False)


if args.last_ep_rec > 0:
    for i in range(dr.sim_len+1):
        end_record_collision[i] = end_record_collision[i]/args.last_ep_rec
        end_record_cat_saved[i] = end_record_cat_saved[i]/args.last_ep_rec

    end_col_ev = pd.DataFrame(np.array(end_record_collision))
    end_col_ev.to_csv('./record_timesteps/{}_{}_{:.2f}_{:.2f}_{}_collisions_end.csv'.format(args.id, args.last_ep_rec, args.temp, args.gamma, label), index=False)
    end_cats_ev = pd.DataFrame(np.array(end_record_cat_saved))
    end_cats_ev.to_csv('./record_timesteps/{}_{}_{:.2f}_{:.2f}_{}_cat_hits_end.csv'.format(args.id, args.last_ep_rec, args.temp, args.gamma, label), index=False)



df = pd.DataFrame(np.array(episode_rewards))
df.to_csv('./record/{}_{:.2f}_{:.2f}_{}_steps.csv'.format(args.id, args.temp, args.gamma, label), index=False)
dfp = pd.DataFrame(np.array(collisions))
dfp.to_csv('./record/{}_{:.2f}_{:.2f}_{}_collisions.csv'.format(args.id, args.cp, args.taup, label), index=False)
dfn = pd.DataFrame(np.array(cat_hits))
dfn.to_csv('./record/{}_{:.2f}_{:.2f}_{}_cat_hits.csv'.format(args.id, args.cn, args.taun, label), index=False)

with open('./policies/'+str(args.id)+'_hpolicy_drive_p.pkl', 'wb') as f:
    pickle.dump(trajectory, f, pickle.HIGHEST_PROTOCOL)
