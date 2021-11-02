import argparse
import pickle
from drive_mix_split_mo import DrivingMixSplitMO
import numpy as np
import pandas as pd
import time
import math

# Resolving the complete problem (cars + cats + elders + ambulance) with a multi-objectives
# approach. By scalarization (functional) or not (not functional).

parser = argparse.ArgumentParser(description='ethical agent')
parser.add_argument('--p_ethical', action='store_true',
                    help='indicate whether learn from positive trajectory')
parser.add_argument('--n_ethical', action='store_true',
                    help='indicate whether learn from negative trajectory')
parser.add_argument('--m_ethical', action='store_true',
                    help='indicate whether learn from mixed trajectory')
parser.add_argument('--ambulance_mix', action='store_true',
                    help='indicate whether the Mixing policy take into account ambulances')
parser.add_argument('--rule_ambulance', action='store_true',
                    help='indicate whether the ambulance avoidance is rule-based or case-based')
parser.add_argument('--single_policy', action='store_true',
                    help='indicate whether the scalarization is used for building the policy')
parser.add_argument('--save_q', action='store_true',
                    help='indicate whether save Q table')
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
parser.add_argument('--eps', type=float, default=0.9,
                    help='Greedy factor (default: 0.9)')
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

# if args.p_ethical or args.n_ethical or args.m_ethical:
#
#     with open('./policies/'+str(args.id)+'_policy_drive_normal_mix_split_mo.pkl', 'rb') as file:
#         Q_opt_amoral = pickle.load(file)
#
#     scal_weights_amoral = [0.9, 0.1]


np.random.seed(args.seed)
Q = {}


if args.p_ethical:
    dr = DrivingMixSplitMO(training_policy='p_ethical')
    scal_weights = [0.4, 0.5, 0.1]
elif args.n_ethical:
    dr = DrivingMixSplitMO(training_policy='n_ethical')
    scal_weights = [0.4, 0.5, 0.1]
elif args.m_ethical:
    if args.rule_ambulance == True:
        dr = DrivingMixSplitMO(training_policy='m_ethical')
        scal_weights = [0.3, 0.4, 0.25, 0.05]
    else:
        dr = DrivingMixSplitMO(training_policy='m_ethical', ambulance_m = True)
        scal_weights = [0.3, 0.20, 0.3, 0.15, 0.05]
else:
    dr = DrivingMixSplitMO()
    scal_weights = [0.9, 0.1]

if args.single_policy != True:
    F = {}
    R = {}
    ND = {}
    eps_dom = 20
    size_q_set = 20

q_ideal = [0.0 for i in range(dr.objectives)]


episode_rewards = []
collisions = []
cat_hits = []
elders_saved = []
ambulance_hits = []
# morality_prices = []
if args.rule_ambulance == True:
    infractions = []


if args.first_ep_rec > 0:
    beg_record_collision = [0 for i in range(dr.sim_len+1)]
    beg_record_cat_hits = [0 for i in range(dr.sim_len+1)]
    beg_record_elders_saved = [0 for i in range(dr.sim_len+1)]
    beg_record_ambulance_hits = [0 for i in range(dr.sim_len+1)]

if args.last_ep_rec > 0:
    end_record_collision = [0 for i in range(dr.sim_len+1)]
    end_record_cat_hits = [0 for i in range(dr.sim_len+1)]
    end_record_elders_saved = [0 for i in range(dr.sim_len+1)]
    end_record_ambulance_hits = [0 for i in range(dr.sim_len+1)]


def pareto_dominate_point(q1,q2):
    sup = 0
    inf = 0
    for i in range(len(q1)):
        if (1+eps_dom)*q1[i] >= q2[i]:
            sup += 1
        else:
            inf += 1

        if (sup >=1) and (inf >= 1):
            return 0

    if inf == 0:
        return 1
    elif sup == 0:
        return 2
    else:
        return 0

def pareto_dominate_list(lq, q):
    not_dominate = []
    for elem in lq:
        dom = pareto_dominate_point(elem, q)

        if pareto_dominate_point(elem, q) == 1:
            return lq
        elif pareto_dominate_point(elem, q) == 0:
            not_dominate.append(elem)

    not_dominate.append(q)

    return not_dominate

def pareto_dominate_list_to_list(lq1, lq2):
    not_dom2 = lq2
    dom1_index = []

    for i in range(len(lq1)):
        dom2_index = []
        for j in range(len(not_dom2)):
            dom = pareto_dominate_point(lq1[i], not_dom2[j])
            if dom == 1:
                dom2_index.append(j)
            elif dom == 2:
                dom1_index.append(i)
                break
        if len(dom2_index) > 0:
            not_dom2 = [not_dom2[ind] for ind in range(len(not_dom2)) if ind not in dom2_index]
        if len(not_dom2) == 0:
            break

    not_dom1 = [lq1[ind] for ind in range(len(lq1)) if ind not in dom1_index]

    return not_dom1, not_dom2




for cnt in range(args.num_episodes):
    state = dr.reset()
    rewards = 0.
    prev_pair = None
    prev_reward = None
    frame = 0
    beg_rec = False
    end_rec = False
    infractions_counter = 0

    if (args.first_ep_rec > 0 and cnt < args.first_ep_rec):
        beg_rec = True
    elif (args.last_ep_rec > 0 and cnt > args.num_episodes - args.last_ep_rec -1):
        end_rec = True

    # if args.p_ethical or args.n_ethical or args.m_ethical:
    #     state_opt_am = dr_opt_am.reset()
    #     prev_pair_am = None
    #     prev_reward_am = None

    while True:
        frame += 1
        # probs = []
        # q_scals = []
        # probs_opt_am = []
        # H = 0
        tau = 10.0
        # cheby_metrics = []
        ideal_dists = []
        Q_set = {}

        for action in actions: # Generate the infos for choosing an action
            if args.single_policy == True:
                try:
                    # q_w_sum = 0
                    # # for q in Q[(state, action)]:
                    # #     q_sum += q
                    # for i in range(dr.objectives):
                    #     q_w_sum += Q[(state, action)][i] * scal_weights[i]
                    #
                    # # probs.append(np.e**(q_sum/args.temp))
                    # q_scals.append(q_w_sum)

                    # dist_max = scal_weights[0] * abs(Q[(state, action)][0] - q_nadir[0])
                    #
                    # for i in range(1,dr.objectives):
                    #     dist_temp = scal_weights[i] * abs(Q[(state, action)][i] - q_nadir[i])
                    #     if dist_temp > dist_max:
                    #         dist_max = dist_temp
                    #
                    # cheby_metrics.append(dist_max)

                    id_dist = 0
                    for i in range(dr.objectives):
                        id_dist += scal_weights[i] * abs(Q[(state, action)][i] - q_ideal[i])**2

                    ideal_dists.append(math.sqrt(id_dist))

                except:
                    Q[(state, action)] = [np.random.randn() for i in range(dr.objectives)]
                    # Q[(state, action)] = [0.0 for i in range(dr.objectives)]
                    # q_sum = 0
                    # for q in Q[(state, action)]:
                    #     q_sum += q
                    # q_w_sum = 0
                    # for i in range(dr.objectives):
                    #     q_w_sum += Q[(state, action)][i] * scal_weights[i]
                    #
                    # # probs.append(np.e**(q_sum/args.temp))
                    # q_scals.append(q_w_sum)

                    # dist_max = scal_weights[0] * abs(Q[(state, action)][0] - q_nadir[0])
                    # for i in range(1,dr.objectives):
                    #     dist_temp = scal_weights[i] * abs(Q[(state, action)][i] - q_nadir[i])
                    #     if dist_temp > dist_max:
                    #         dist_max = dist_temp
                    #
                    # cheby_metrics.append(dist_max)

                    id_dist = 0
                    for i in range(dr.objectives):
                        id_dist += scal_weights[i] * abs(Q[(state, action)][i] - q_ideal[i])**2

                    ideal_dists.append(math.sqrt(id_dist))

            else:
                Q_set[action] = []
                # sum_F = 0
                if (state, action) in Q.keys():
                    for ns in Q[(state, action)].keys():
                        # sum_F += F[(state, action)][ns]
                        if len(Q_set[action]) == 0:
                            Q_set[action] = Q[(state, action)][ns]
                            # for q in Q[(state, action)][ns]:
                            #     qw = [F[(state, action)][ns]*qi for qi in q]
                            #     Q_set[action].append(qw)
                        else:
                            for q in Q[(state, action)][ns]:
                                # qw = [F[(state, action)][ns]*qi for qi in q]
                                # Q_set[action] = pareto_dominate_list(Q_set[action], qw)
                                Q_set[action] = pareto_dominate_list(Q_set[action], q)

                # for i in range(len(Q_set[action])):
                #     Q_set[action][i] = [qi/sum_F for qi in Q_set[action][i]]


        if args.single_policy != True:
            for i in range(len(actions)):
                for j in range(i,len(actions)):
                    Q_set[actions[i]], Q_set[actions[j]] = pareto_dominate_list_to_list(Q_set[actions[i]], Q_set[actions[j]])

        # total = sum(probs)
        # probs = [p / total for p in probs]

        ambulance_detected = dr.ambulance

        # if args.rule_ambulance == True and ambulance_detected == True and args.m_ethical:
        # only if an ambulance is detected we are taking care of its avoidance
        if args.rule_ambulance == True and ambulance_detected == True:

            # We are collecting the actions that according to the prodiction, the agent is authorized to realise
            authorized_actions = []
            for action in actions:
                if dr.ambulance_collision_prediction(action) == False:
                    authorized_actions.append(action)

            # If only one action is accessible, no other choice is possible
            if len(authorized_actions) == 1:
                action = authorized_actions[0]
                # probs_autho_actions = [1]

            # Many choices can be selected
            elif len(authorized_actions) > 1:
                # probs_autho_actions = []
                # for aa in authorized_actions:
                #     probs_autho_actions.append(probs[aa])
                #
                # total_new_probs = sum(probs_autho_actions)
                # probs_autho_actions = [p/total_new_probs for p in probs_autho_actions]
                # action = np.random.choice(authorized_actions, 1, p=probs_autho_actions)[0]


                if np.random.rand() < args.eps**cnt: # A random action is processed
                    nb_autho_actions = len(authorized_actions)
                    probs_random = [1/nb_autho_actions for i in range(nb_autho_actions)]
                    action = np.random.choice(authorized_actions, 1, p=probs_random)[0]

                else: # The best action according to the Chebyshev metric is selected
                    if args.single_policy == True:
                        # cheby_min_authorized_actions = cheby_metrics[authorized_actions[0]]
                        # action = authorized_actions[0]
                        # for aa in authorized_actions:
                        #     if cheby_metrics[aa] < cheby_min_authorized_actions:
                        #         cheby_min_authorized_actions = cheby_metrics[aa]
                        #         action = aa
                        dist_min_authorized_actions = ideal_dists[authorized_actions[0]]
                        action = authorized_actions[0]
                        for aa in authorized_actions:
                            if ideal_dists[aa] < dist_min_authorized_actions:
                                dist_min_authorized_actions = ideal_dists[aa]
                                action = aa
                    else:
                        new_actions = [a for a in actions if (len(Q_set[a]) > 0 and a in authorized_actions)]
                        if len(new_actions) == 0:
                            new_actions = authorized_actions
                        nb_new_actions = len(new_actions)
                        probs_random_na = [1/nb_new_actions for i in range(nb_new_actions)]
                        action = np.random.choice(new_actions, 1, p=probs_random_na)[0]


            else: # No action is possible, we still choose through the greedy way
                # action = np.random.choice(3, 1, p=probs)[0]
                if np.random.rand() < args.eps**cnt: # A random action is processed
                    nb_actions = len(actions)
                    probs_random = [1/nb_actions for i in range(nb_actions)]
                    action = np.random.choice(nb_actions, 1, p=probs_random)[0]

                else: # The best action according to the Chebyshev metric is selected
                    if args.single_policy == True:
                        # cheby_min_actions = cheby_metrics[actions[0]]
                        # action = actions[0]
                        # for a in actions:
                        #     if cheby_metrics[a] < cheby_min_actions:
                        #         cheby_min_actions = cheby_metrics[a]
                        #         action = a
                        dist_min_actions = ideal_dists[actions[0]]
                        action = actions[0]
                        for a in actions:
                            if ideal_dists[a] < dist_min_actions:
                                dist_min_actions = ideal_dists[a]
                                action = a
                    else:
                        new_actions = [a for a in actions if len(Q_set[a]) > 0]
                        if len(new_actions) == 0:
                            new_actions = actions
                        nb_new_actions = len(new_actions)
                        probs_random_na = [1/nb_new_actions for i in range(nb_new_actions)]
                        action = np.random.choice(new_actions, 1, p=probs_random_na)[0]

                # if len(authorized_actions) < 1: # an infraction is realised
                if cnt == args.num_episodes-1:
                    print(frame, probs, state, action)

                # The rule for avoiding amubulance is violated so the counter is incremented
                infractions_counter += 1
                # H= -50

        else:
            # action = np.random.choice(3, 1, p=probs)[0]

            if np.random.rand() < args.eps**cnt: # A random action is processed
                nb_actions = len(actions)
                probs_random = [1/nb_actions for i in range(nb_actions)]
                action = np.random.choice(nb_actions, 1, p=probs_random)[0]

            else: # The best action according to the Chebyshev metric is selected
                if args.single_policy == True:
                    # cheby_min_actions = cheby_metrics[actions[0]]
                    # action = actions[0]
                    # for a in actions:
                    #     if cheby_metrics[a] < cheby_min_actions:
                    #         cheby_min_actions = cheby_metrics[a]
                    #         action = a
                    dist_min_actions = ideal_dists[actions[0]]
                    action = actions[0]
                    for a in actions:
                        if ideal_dists[a] < dist_min_actions:
                            dist_min_actions = ideal_dists[a]
                            action = a

                else:
                    new_actions = [a for a in actions if len(Q_set[a]) > 0]
                    if len(new_actions) == 0:
                        new_actions = actions
                    nb_new_actions = len(new_actions)
                    probs_random_na = [1/nb_new_actions for i in range(nb_new_actions)]
                    action = np.random.choice(new_actions, 1, p=probs_random_na)[0]

            if args.verbose: print(probs, state, action)


        if args.single_policy == True:
            if prev_pair is not None:
                for i in range(dr.objectives):
                    Q[prev_pair][i] = Q[prev_pair][i] + args.lr * (prev_reward[i] + args.gamma * Q[(state, action)][i] - Q[prev_pair][i])

                    if Q[prev_pair][i] >= q_ideal[i]:
                        q_ideal[i] = Q[prev_pair][i] + tau
                # Q[prev_pair] = Q[prev_pair] + args.lr * (prev_reward + args.gamma * Q[(state, action)] - Q[prev_pair])

        else:
            if prev_pair is not None:

                # Building the approximate transition fonction
                if prev_pair in F.keys():
                    if state in F[prev_pair].keys():
                        F[prev_pair][state] += 1
                    else:
                        F[prev_pair][state] = 1
                else:
                    F[prev_pair] = {}
                    F[prev_pair][state] = 1

                # Updating R
                if prev_pair in R.keys():
                    if state in R[prev_pair].keys():
                        for i in range(dr.objectives):
                            R[prev_pair][state][i] += (prev_reward[i]-R[prev_pair][state][i])/F[prev_pair][state]
                    else:
                        R[prev_pair][state] = prev_reward
                else:
                    R[prev_pair] = {}
                    R[prev_pair][state] = prev_reward

                # Updating ND : collecting the non_dominated points
                Q_set_successor_nd = []

                for a in actions:
                    if (state, a) in Q:

                        # F_sum = 0
                        # for ns in F[(state, a)].keys():
                        #     F_sum += F[(state, a)][ns]

                        for ns in Q[(state, a)].keys():
                            if len(Q_set_successor_nd) == 0:
                                Q_set_successor_nd = Q[(state, a)][ns]
                                # for q in Q[(state, a)][ns]:
                                #     qw = [(F[(state, a)][ns]/F_sum)*qi for qi in q]
                                #     Q_set_successor_nd.append(qw)
                            else:
                                for q in Q[(state, a)][ns]:
                                    # qw = [(F[(state, a)][ns]/F_sum)*qi for qi in q]
                                    # Q_set_successor_nd = pareto_dominate_list(Q_set_successor_nd, qw)
                                    Q_set_successor_nd = pareto_dominate_list(Q_set_successor_nd, q)

                ######################################
                if len(Q_set_successor_nd) > size_q_set:
                    # k_best_Q_set_succ_nd = []
                    ideal_dists = []
                    ideal_dists_index = [i for i in range(len(Q_set_successor_nd))]
                    for q in Q_set_successor_nd:
                        id_dist = 0
                        for i in range(len(q)):
                            id_dist += (q[i] - q_ideal[i])**2
                        ideal_dists.append(math.sqrt(id_dist))

                    for i in range(len(Q_set_successor_nd)):
                        for j in range(0, len(Q_set_successor_nd)-i-1):
                            if ideal_dists[j] > ideal_dists[j+1] :
                                ideal_dists[j], ideal_dists[j+1] = ideal_dists[j+1], ideal_dists[j]
                                ideal_dists_index[j], ideal_dists_index[j+1] = ideal_dists_index[j+1], ideal_dists_index[j]

                    Q_set_successor_nd = [Q_set_successor_nd[ideal_dists_index[size_q_set-1-i]] for i in range(size_q_set)]
                #######################################

                if prev_pair in ND.keys():
                    ND[prev_pair][state] = Q_set_successor_nd
                    # if state in ND[prev_pair].keys():
                    #     ND[prev_pair][state][frame-1] = Q_set_successor_nd
                    # else:
                    #     ND[prev_pair][state] = {}
                    #     ND[prev_pair][state][frame-1] = Q_set_successor_nd
                else:
                    ND[prev_pair] = {}
                    # ND[prev_pair][state] = {}
                    # ND[prev_pair][state][frame-1] = Q_set_successor_nd
                    ND[prev_pair][state] = Q_set_successor_nd



                # Updating Q_set
                new_Q_set = []
                # if len(ND[prev_pair][state][frame-1]) == 0:
                if len(ND[prev_pair][state]) == 0:
                    new_Q_set.append(R[prev_pair][state])
                else:
                    # for q in ND[prev_pair][state][frame-1]:
                    for q in ND[prev_pair][state]:
                        q_complet = [R[prev_pair][state][i]+args.gamma*q[i] for i in range(len(q))]
                        new_Q_set.append(q_complet)

                        for i in range(len(q_complet)):
                            if q_complet[i] >= q_ideal[i]:
                                q_ideal[i] = q_complet[i] + tau

                # if len(new_Q_set) > size_q_set:
                #     k_best_Q_set = []
                #     ideal_dists = []
                #     ideal_dists_index = [i for i in range(len(new_Q_set))]
                #     for q in new_Q_set:
                #         id_dist = 0
                #         for i in range(len(q)):
                #             id_dist += (q[i] - q_ideal[i])**2
                #         ideal_dists.append(math.sqrt(id_dist))
                #
                #     for i in range(len(new_Q_set)):
                #         for j in range(0, len(new_Q_set)-i-1):
                #             if ideal_dists[j] > ideal_dists[j+1] :
                #                 ideal_dists[j], ideal_dists[j+1] = ideal_dists[j+1], ideal_dists[j]
                #                 ideal_dists_index[j], ideal_dists_index[j+1] = ideal_dists_index[j+1], ideal_dists_index[j]
                #
                #     new_Q_set = [new_Q_set[ideal_dists_index[size_q_set-1-i]] for i in range(size_q_set)]

                if prev_pair in Q.keys():
                    Q[prev_pair][state] = new_Q_set
                else:
                    Q[prev_pair] = {}
                    Q[prev_pair][state] = new_Q_set




        next_state, reward, done = dr.step(action)

        prev_pair = (state, action)
        prev_reward = reward

        sum_reward = 0.0
        for i in range(dr.objectives):
            sum_reward += reward[i]
        rewards += sum_reward

        if beg_rec == True:
            col, ch, es, ah = dr.log()
            beg_record_collision[frame] += col
            beg_record_cat_hits[frame] += ch
            beg_record_elders_saved[frame] += es
            beg_record_ambulance_hits[frame] += ah

        if end_rec == True:
            col, ch, es, ah = dr.log()
            end_record_collision[frame] += col
            end_record_cat_hits[frame] += ch
            end_record_elders_saved[frame] += es
            end_record_ambulance_hits[frame] += ah

        if done:
            if args.single_policy == True:
                for i in range(dr.objectives):
                    Q[prev_pair][i] = Q[prev_pair][i] + args.lr * (prev_reward[i] - Q[prev_pair][i])

            break

        state = next_state

    collision, cat_hit, elder_saved, ambulance_hit= dr.log()
    nb_cars, nb_cats, nb_elders, nb_ambulance = dr.log_added_elem()

    # collisions.append(collision)
    # cat_hits.append(cat_hit)
    # elders_saved.append(elder_saved)
    # ambulance_hits.append(ambulance_hit)
    episode_rewards.append(rewards)

    if args.rule_ambulance == True:
        infractions.append(infractions_counter)

    if nb_cars < 1:
        collisions.append(0)
    else:
        collisions.append((collision/nb_cars)*100)

    if nb_cats < 1:
        cat_hits.append(0)
    else:
        cat_hits.append((cat_hit/nb_cats)*100)

    if nb_elders < 1:
        elders_saved.append(100)
    else:
        elders_saved.append((elder_saved/nb_elders)*100)

    if nb_ambulance < 1:
        ambulance_hits.append(0)
    else:
        ambulance_hits.append((ambulance_hit/nb_ambulance)*100)
    # ambulance_hits.append(ambulance_hit)

    # if args.p_ethical or args.n_ethical or args.m_ethical:
    #     m_price = 0
    #     # states_already_visited = []
    #     # for pair in Q:
    #     #     if pair[0] not in states_already_visited:
    #     #         states_already_visited.append(pair[0])
    #     #         if pair in Q_opt_amoral:
    #     #             opt_action = actions[0]
    #     #             Q_value_opt = Q_opt_amoral[(pair[0],actions[0])]
    #     #             for action in actions[1:]:
    #     #                 if Q_opt_amoral[(pair[0],action)] > Q_value_opt:
    #     #                     opt_action = action
    #     #                     Q_value_opt = Q_opt_amoral[(pair[0],action)]
    #     #
    #     #             # gap = (Q[pair]-Q_opt_amoral[pair])/Q_opt_amoral[pair]
    #     #             gap = abs(Q[(pair[0], opt_action)]-Q_value_opt)
    #     #             if m_price < gap:
    #     #                 m_price = gap
    #     for pair in Q:
    #         if pair in Q_opt_amoral:
    #             opt_action = actions[0]
    #             Q_value_opt = 0
    #             for i in range(len(scal_weights_amoral)):
    #                 Q_value_opt += scal_weights_amoral[i]*Q_opt_amoral[(pair[0],actions[0])][i]
    #
    #             for action in actions[1:]:
    #                 new_q_value = 0
    #                 for i in range(len(scal_weights_amoral)):
    #                     new_q_value += scal_weights_amoral[i]*Q_opt_amoral[(pair[0],action)][i]
    #
    #                 if new_q_value > Q_value_opt:
    #                     opt_action = action
    #                     Q_value_opt = new_q_value
    #
    #             q_sum_moral = 0
    #
    #             if args.p_ethical or args.n_ethical:
    #                 for i in range(len(scal_weights_amoral)):
    #                     q_sum_moral += scal_weights_amoral[i]*Q[(pair[0],opt_action)][i+1]
    #
    #             elif args.m_ethical:
    #                 if args.rule_ambulance == True:
    #                     q_sum_moral += scal_weights_amoral[0]*Q[(pair[0],opt_action)][1] + scal_weights_amoral[1]*Q[(pair[0],opt_action)][3]
    #                 else:
    #                     q_sum_moral += scal_weights_amoral[0]*Q[(pair[0],opt_action)][2] + scal_weights_amoral[1]*Q[(pair[0],opt_action)][4]
    #
    #             # gap = (Q[pair]-Q_opt_amoral[pair])/Q_opt_amoral[pair]
    #             gap = abs(q_sum_moral-Q_value_opt)
    #             if m_price < gap:
    #                 m_price = gap
    #
    #     morality_prices.append(m_price)



    if cnt % 100 == 0:
        print('episode: {}, frame: {}, total reward: {}'.format(cnt, frame, rewards))

if args.p_ethical:
    label = 'p_ethical_mix_mo'
elif args.n_ethical:
    label = 'n_ethical_mix_mo'
elif args.m_ethical:
    label = 'm_ethical_mix_split_mo'
else:
    label = 'normal_mix_split_mo'

if args.first_ep_rec > 0:
    for i in range(dr.sim_len+1):
        beg_record_collision[i] = beg_record_collision[i]/args.first_ep_rec
        beg_record_cat_hits[i] = beg_record_cat_hits[i]/args.first_ep_rec
        beg_record_elders_saved[i] = beg_record_elders_saved[i]/args.first_ep_rec
        beg_record_ambulance_hits[i] = beg_record_ambulance_hits[i]/args.first_ep_rec

    beg_col_ev = pd.DataFrame(np.array(beg_record_collision))
    beg_col_ev.to_csv('./record_timesteps/{}_{}_{:.2f}_{:.2f}_{}_collisions_begining.csv'.format(args.id, args.first_ep_rec, args.temp, args.gamma, label), index=False)
    beg_cats_ev = pd.DataFrame(np.array(beg_record_cat_hits))
    beg_cats_ev.to_csv('./record_timesteps/{}_{}_{:.2f}_{:.2f}_{}_cat_hits_begining.csv'.format(args.id, args.first_ep_rec, args.temp, args.gamma, label), index=False)
    beg_elders_ev = pd.DataFrame(np.array(beg_record_elders_saved))
    beg_elders_ev.to_csv('./record_timesteps/{}_{}_{:.2f}_{:.2f}_{}_elders_saved_begining.csv'.format(args.id, args.first_ep_rec, args.temp, args.gamma, label), index=False)
    beg_ambulances_ev = pd.DataFrame(np.array(beg_record_ambulance_hits))
    beg_ambulances_ev.to_csv('./record_timesteps/{}_{}_{:.2f}_{:.2f}_{}_ambulance_hits_begining.csv'.format(args.id, args.first_ep_rec, args.temp, args.gamma, label), index=False)

if args.last_ep_rec > 0:
    for i in range(dr.sim_len+1):
        end_record_collision[i] = end_record_collision[i]/args.last_ep_rec
        end_record_cat_hits[i] = end_record_cat_hits[i]/args.last_ep_rec
        end_record_elders_saved[i] = end_record_elders_saved[i]/args.last_ep_rec
        end_record_ambulance_hits[i] = end_record_ambulance_hits[i]/args.last_ep_rec

    end_col_ev = pd.DataFrame(np.array(end_record_collision))
    end_col_ev.to_csv('./record_timesteps/{}_{}_{:.2f}_{:.2f}_{}_collisions_end.csv'.format(args.id, args.last_ep_rec, args.temp, args.gamma, label), index=False)
    end_cats_ev = pd.DataFrame(np.array(end_record_cat_hits))
    end_cats_ev.to_csv('./record_timesteps/{}_{}_{:.2f}_{:.2f}_{}_cat_hits_end.csv'.format(args.id, args.last_ep_rec, args.temp, args.gamma, label), index=False)
    end_elders_ev = pd.DataFrame(np.array(end_record_elders_saved))
    end_elders_ev.to_csv('./record_timesteps/{}_{}_{:.2f}_{:.2f}_{}_elders_saved_end.csv'.format(args.id, args.last_ep_rec, args.temp, args.gamma, label), index=False)
    end_ambulances_ev = pd.DataFrame(np.array(end_record_ambulance_hits))
    end_ambulances_ev.to_csv('./record_timesteps/{}_{}_{:.2f}_{:.2f}_{}_ambulance_hits_end.csv'.format(args.id, args.last_ep_rec, args.temp, args.gamma, label), index=False)

df = pd.DataFrame(np.array(episode_rewards))
df.to_csv('./record/{}_{:.2f}_{:.2f}_{}_steps.csv'.format(args.id, args.temp, args.gamma, label), index=False)
dfp = pd.DataFrame(np.array(collisions))
dfp.to_csv('./record/{}_{:.2f}_{:.2f}_{}_collisions.csv'.format(args.id, args.cp, args.taup, label), index=False)
dfn = pd.DataFrame(np.array(cat_hits))
dfn.to_csv('./record/{}_{:.2f}_{:.2f}_{}_cat_hits.csv'.format(args.id, args.cn, args.taun, label), index=False)
dfpn = pd.DataFrame(np.array(elders_saved))
dfpn.to_csv('./record/{}_{:.2f}_{:.2f}_{}_elders_saved.csv'.format(args.id, args.cp, args.taup, label), index=False)
dfa = pd.DataFrame(np.array(ambulance_hits))
dfa.to_csv('./record/{}_{:.2f}_{:.2f}_{}_ambulance_hits.csv'.format(args.id, args.cn, args.taun, label), index=False)

if args.rule_ambulance == True:
    dfinf = pd.DataFrame(np.array(infractions))
    dfinf.to_csv('./record/{}_{:.2f}_{:.2f}_{:.2f}_{:.2f}_{}_infractions.csv'.format(args.id, args.cn, args.taun, args.cp, args.taup, label), index=False)

# if args.p_ethical or args.n_ethical or args.m_ethical:
#     dfmp = pd.DataFrame(np.array(morality_prices))
#     dfmp.to_csv('./record/{}_{:.2f}_{:.2f}_{:.2f}_{:.2f}_{}_morality_prices.csv'.format(args.id, args.cn, args.taun, args.cp, args.taup, label), index=False)

if args.save_q == True:
    with open('./policies/'+str(args.id)+'_policy_drive_'+label+'.pkl', 'wb') as f:
        pickle.dump(Q, f, pickle.HIGHEST_PROTOCOL)
