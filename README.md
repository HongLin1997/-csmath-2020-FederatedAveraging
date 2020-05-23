# Federated Averaging for csmath-2020 course project

#### Sample command for K=0.1 N=100 E=1 B=10 (B is the local batch size) dataset=mnist

    python -u main_fed.py \
    --epoch=200 \
    --model=softmax \
    --dataset=mnist \
    --noniid=1 \
    --class_per_device=2 \
    --gpu=1 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=1 \
    --local_bs=10 \
    --lr=0.01 \
    --decay_rate=0.995 \
    --per_epoch=1 \
    --num_channels=1 \
    --unbalance=2.0
    
#### Sample command K=0.2 N=100 E=5 B=10 (B is the local batch size) dataset=cifar 

    python -u main_fed.py \
    --epoch=200 \
    --model=softmax \
    --dataset=cifar \
    --noniid=1 \
    --class_per_device=2 \
    --gpu=1 \
    --num_users=100 \
    --frac=0.2 \
    --local_ep=5 \
    --local_bs=10 \
    --lr=0.01 \
    --decay_rate=0.992 \
    --per_epoch=1 \
    --num_channels=3 \
    --unbalance=2.0
    
    
    
    
check the **utils/options.py** for the meaning of each argument to ./main_fed.py
