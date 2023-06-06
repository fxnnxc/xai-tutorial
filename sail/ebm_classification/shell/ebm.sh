
config='configs/ebm_'$data'.yaml' 
data_path='untracked'
seed=1
save_interval=1000 # batch 

python ebm_pkg/ebm/train_ebm.py \
    --config $config \
    --model $model \
    --data $data \
    --data-path $data_path \
    --save-interval $save_interval \
    --seed $seed  

    