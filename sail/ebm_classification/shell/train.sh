data_path='untracked'
config='configs/classifier_'$data$'.yaml'
model='cnn'
ebm_path='results/ebm/'$data/'cnn/seed_1'

python train_classifier.py \
    --config $config \
    --model $model \
    --ebm-path $ebm_path \
    --data $data \
    --data-path $data_path \
    --loss $loss \
    --train-type $train_type
    