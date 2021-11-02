#!/bin/sh


for i in `seq 0 99`;
do
        echo $i
        let "seed = 1234 + i"
        echo $seed

        python sarsa_mix.py --save_q --id $i --seed $seed --cn 1.00 --cp 2.00 --gamma 0.99 --num_episodes 4000 --taun 0.20 --taup 0.50
        # python sarsa.py --id $i --seed $seed --cn 1.00 --cp 2.00 --gamma 0.99 --num_episodes 4000 --taun 0.15 --taup 0.50 --first_ep_rec 300 --last_ep_rec 300
        # python sarsa_mix_split.py --save_q --id $i --seed $seed --cn 1.00 --cp 2.00 --gamma 0.99 --num_episodes 4000 --taun 0.20 --taup 0.50 --first_ep_rec 300 --last_ep_rec 300
        # python sarsa_mix_split_mo.py --save_q --id $i --seed $seed --cn 1.00 --cp 2.00 --gamma 0.99 --num_episodes 4000 --taun 0.20 --taup 0.50 --first_ep_rec 300 --last_ep_rec 300
        # python sarsa_mix_split_mo.py --save_q --id $i --seed $seed --cn 1.00 --cp 2.00 --gamma 0.99 --num_episodes 4000 --taun 0.20 --taup 0.50

done
