data_path='untracked'
model='cnn'
train_type='normal'
loss='normal'

for data in mnist fashion_mnist
do 
    config='configs/classifier_'$data$'.yaml'
    ebm_path='results/ebm/'$data/'cnn/seed_1'
    for seed in 0 1 2
    do 
        python train_classifier.py \
            --config $config \
            --model $model \
            --ebm-path $ebm_path \
            --data $data \
            --data-path $data_path \
            --loss $loss \
            --train-type $train_type \
            --seed $seed      
    done 
done 