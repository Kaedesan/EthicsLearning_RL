import argparse
import pickle
from drive_mix import DrivingMix
import numpy as np
import pandas as pd
import time

# Instead of learning from a mixed policy, used both the negative and positive
# one in the reward shapping, and the policies obtained by hsarsa_mix_v2.py
# (= Different ethical state for both policies, depending on what they are
# putting  attention)

parser = argparse.ArgumentParser(description='ethical agent')
parser.add_argument('--p_ethical', action='store_true',
                    help='indicate whether learn from positive trajectory')
parser.add_argument('--n_ethical', action='store_true',
                    help='indicate whether learn from negative trajectory')
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
parser.add_argument('--first_ep_rec', type=int, default= 0,
                    help='number of recorded episodes from the begining')
parser.add_argument('--last_ep_rec', type=int, default= 0,
                    help='number of recorded episodes from the end')
args = parser.parse_args()

actions = range(3)

n_eth = False
p_eth = False

if args.m_ethical:
    n_eth = True
    p_eth = True
elif args.p_ethical:
    p_eth = True
elif args.n_ethical:
    n_eth = True

if n_eth == True:
    hnpolicy = {}
    hn_filename = './policies/'+str(args.id)+'_hpolicy_drive_human_n_mix.pkl'
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
                count = [p * (args.count_scale / total_cnt) for p in count]

            total_cnt = sum(count)
            probs = [args.c**count[action]*(1-args.c)**(total_cnt-count[action]) for action in actions]
            print(probs)
            total_prob = sum(probs)
            probs = [p / total_prob for p in probs]
            hnpolicy[key[0]] = probs

if p_eth == True:
    hppolicy = {}
    hp_filename = './policies/'+str(args.id)+'_hpolicy_drive_human_p_mix.pkl'

    with open(hp_filename, 'rb') as f:
        hp_trajectory = pickle.load(f)

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


np.random.seed(args.seed)
Q = {}
if args.p_ethical:
    dr = DrivingMix(training_policy='p_ethical')
elif args.n_ethical:
    dr = DrivingMix(training_policy='n_ethical')
elif args.m_ethical:
    dr = DrivingMix(training_policy='m_ethical')
else:
    dr = DrivingMix()

episode_rewards = []
collisions = []
cat_hits = []
elders_saved = []

if args.first_ep_rec > 0:
    beg_record_collision = [0 for i in range(dr.sim_len+1)]
    beg_record_cat_hits = [0 for i in range(dr.sim_len+1)]
    beg_record_elders_saved = [0 for i in range(dr.sim_len+1)]

if args.last_ep_rec > 0:
    end_record_collision = [0 for i in range(dr.sim_len+1)]
    end_record_cat_hits = [0 for i in range(dr.sim_len+1)]
    end_record_elders_saved = [0 for i in range(dr.sim_len+1)]

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

        if prev_pair is not None:
            Q[prev_pair] = Q[prev_pair] + args.lr * (prev_reward + args.gamma * Q[(state, action)] - Q[prev_pair])
        next_state, reward, done = dr.step(action)

        H = 0
        if n_eth:
            if args.m_ethical:
                ethical_state_hn = (state[2], state[5], state[8], state[10])
            else:
                ethical_state_hn = (state[2], state[4], state[6], state[7])

            if ethical_state_hn in hnpolicy:
                hnprobs = hnpolicy[ethical_state_hn]
                if hnprobs[action] < args.taun and hnprobs[action] < probs[action]: # Only negative part of the reward shaping?? = Forbide acts
                    H += -args.cn * kl_div(probs, hnprobs)
                    #print("Neg reward shapping = ", H, " at time step ", dr.timestamp)
        if p_eth:
            if args.m_ethical:
                ethical_state_hp = (state[3], state[6], state[9])
            else:
                ethical_state_hp = (state[2], state[4], state[6])

            if ethical_state_hp in hppolicy:
                hpprobs = hppolicy[ethical_state_hp]
                if hpprobs[action] > args.taup and hpprobs[action] > probs[action]: # Only positive part of the reward shaping?? = Push to act
                    H += args.cp * kl_div(probs, hpprobs)
                    #print("Pos reward shapping = ", H, " at time step ", dr.timestamp)
        reward += H

        prev_pair = (state, action)
        prev_reward = reward
        rewards += (reward - H)

        if beg_rec == True:
            col, ch, es= dr.log()
            beg_record_collision[dr.timestamp] += col
            beg_record_cat_hits[dr.timestamp] += ch
            beg_record_elders_saved[dr.timestamp] += es
        if end_rec == True:
            col, ch, es= dr.log()
            end_record_collision[dr.timestamp] += col
            end_record_cat_hits[dr.timestamp] += ch
            end_record_elders_saved[dr.timestamp] += es

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


if args.p_ethical:
    label = 'p_ethical'
elif args.n_ethical:
    label = 'n_ethical'
elif args.m_ethical:
    label = 'mix_ethical'
else:
    label = 'normal'

if args.first_ep_rec > 0:
    for i in range(dr.sim_len+1):
        beg_record_collision[i] = beg_record_collision[i]/args.first_ep_rec
        beg_record_cat_hits[i] = beg_record_cat_hits[i]/args.first_ep_rec
        beg_record_elders_saved[i] = beg_record_elders_saved[i]/args.first_ep_rec

    beg_col_ev = pd.DataFrame(np.array(beg_record_collision))
    beg_col_ev.to_csv('./record_timesteps/{}_{}_{:.2f}_{:.2f}_{}_collisions_begining.csv'.format(args.id, args.first_ep_rec, args.temp, args.gamma, label), index=False)
    beg_cats_ev = pd.DataFrame(np.array(beg_record_cat_hits))
    beg_cats_ev.to_csv('./record_timesteps/{}_{}_{:.2f}_{:.2f}_{}_cat_hits_begining.csv'.format(args.id, args.first_ep_rec, args.temp, args.gamma, label), index=False)
    beg_elders_ev = pd.DataFrame(np.array(beg_record_elders_saved))
    beg_elders_ev.to_csv('./record_timesteps/{}_{}_{:.2f}_{:.2f}_{}_elders_saved_begining.csv'.format(args.id, args.first_ep_rec, args.temp, args.gamma, label), index=False)

if args.last_ep_rec > 0:
    for i in range(dr.sim_len+1):
        end_record_collision[i] = end_record_collision[i]/args.last_ep_rec
        end_record_cat_hits[i] = end_record_cat_hits[i]/args.last_ep_rec
        end_record_elders_saved[i] = end_record_elders_saved[i]/args.last_ep_rec

    end_col_ev = pd.DataFrame(np.array(end_record_collision))
    end_col_ev.to_csv('./record_timesteps/{}_{}_{:.2f}_{:.2f}_{}_collisions_end.csv'.format(args.id, args.last_ep_rec, args.temp, args.gamma, label), index=False)
    end_cats_ev = pd.DataFrame(np.array(end_record_cat_hits))
    end_cats_ev.to_csv('./record_timesteps/{}_{}_{:.2f}_{:.2f}_{}_cat_hits_end.csv'.format(args.id, args.last_ep_rec, args.temp, args.gamma, label), index=False)
    end_elders_ev = pd.DataFrame(np.array(end_record_elders_saved))
    end_elders_ev.to_csv('./record_timesteps/{}_{}_{:.2f}_{:.2f}_{}_elders_saved_end.csv'.format(args.id, args.last_ep_rec, args.temp, args.gamma, label), index=False)


df = pd.DataFrame(np.array(episode_rewards))
df.to_csv('./record/{}_{:.2f}_{:.2f}_{}_steps.csv'.format(args.id, args.temp, args.gamma, label), index=False)
dfp = pd.DataFrame(np.array(collisions))
dfp.to_csv('./record/{}_{:.2f}_{:.2f}_{}_collisions.csv'.format(args.id, args.cp, args.taup, label), index=False)
dfn = pd.DataFrame(np.array(cat_hits))
dfn.to_csv('./record/{}_{:.2f}_{:.2f}_{}_cat_hits.csv'.format(args.id, args.cn, args.taun, label), index=False)
dfpn = pd.DataFrame(np.array(elders_saved))
dfpn.to_csv('./record/{}_{:.2f}_{:.2f}_{}_elders_saved.csv'.format(args.id, args.cp, args.taup, label), index=False)
