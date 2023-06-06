

data=mnist 
for seed in 0 1 2 
do 
    python run.py --data $data --seed $seed --data-path untracked
done 