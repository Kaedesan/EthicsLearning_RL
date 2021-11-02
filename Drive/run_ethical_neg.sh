#!/bin/sh


for i in `seq 0 99`;
do
        echo $i
        let "seed = 1234 + i"
        echo $seed

        # python sarsa_mix_v3.py --n_ethical --id $i --seed $seed --cn 1.00 --cp 2.00 --gamma 0.99 --num_episodes 4000 --taun 0.20 --taup 0.50 --count_scale 40 --first_ep_rec 300 --last_ep_rec 300
        python sarsa_mix.py --n_ethical --id $i --seed $seed --cn 1.00 --cp 2.00 --gamma 0.99 --num_episodes 4000 --taun 0.20 --taup 0.50
        #python sarsa.py --n_ethical --id $i --seed $seed --cn 1.00 --cp 2.00 --gamma 0.99 --num_episodes 4000 --taun 0.20 --taup 0.50 --first_ep_rec 300 --last_ep_rec 300
        # python sarsa_mix_split.py --n_ethical --rule_ambulance --id $i --seed $seed --cn 1.00 --cp 2.00 --gamma 0.99 --num_episodes 4000 --taun 0.20 --taup 0.50 --first_ep_rec 300 --last_ep_rec 300
        # python sarsa_mix_split.py --n_ethical --rule_ambulance --id $i --seed $seed --cn 1.00 --cp 2.00 --gamma 0.99 --num_episodes 4000 --taun 0.20 --taup 0.50 --first_ep_rec 300 --last_ep_rec 300
        # python sarsa_mix_split_mo.py --n_ethical --rule_ambulance --id $i --seed $seed --cn 1.00 --cp 2.00 --gamma 0.99 --num_episodes 4000 --taun 0.20 --taup 0.50 --first_ep_rec 300 --last_ep_rec 300
done
