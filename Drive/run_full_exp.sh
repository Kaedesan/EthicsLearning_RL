#!/bin/sh


for i in `seq 0 29`;
do
        echo $i
        let "seed = 1234 + i"
        echo $seed

        python hsarsa_mix.py --n_ethical --id $i --seed $seed --cn 1.00 --cp 2.00 --gamma 0.99 --num_episodes 4000 --record_beg 2400

        python hsarsa_mix.py --p_ethical --id $i --seed $seed --cn 1.00 --cp 2.00 --gamma 0.99 --num_episodes 4000 --record_beg 2400

        #python hsarsa_mix.py --m_ethical --id $i --seed $seed --cn 1.00 --cp 2.00 --gamma 0.99 --num_episodes 4000 --record_beg 2400

        python sarsa_mix.py --n_ethical --id $i --seed $seed --cn 1.00 --cp 2.00 --gamma 0.99 --num_episodes 4000

        python sarsa_mix.py --p_ethical --id $i --seed $seed --cn 1.00 --cp 2.00 --gamma 0.99 --num_episodes 4000

        #python sarsa_mix_v2.py --m_ethical --id $i --seed $seed --cn 1.00 --cp 2.00 --gamma 0.99 --num_episodes 4000
        python sarsa_mix_v2.py --m_ethical --id $i --seed $seed --cn 1.00 --cp 2.00 --gamma 0.99 --num_episodes 4000

        python sarsa_mix.py --id $i --seed $seed --cn 1.00 --cp 2.00 --gamma 0.99 --num_episodes 4000
done
