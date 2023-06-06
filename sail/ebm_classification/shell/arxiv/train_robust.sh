data_path='untracked'
model='cnn'
train_type='robust'
loss='normal'

for data in mnist fashion_mnist
do 
    ebm_path='results/ebm/'$data/'cnn/seed_1'
    config='configs/classifier_'$data$'.yaml'
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