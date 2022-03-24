# alexnet
python main.py \
    --model alexnet \
    --batch_size 128 \
    --num_epoch 10 \
    --dataset d5 \
    --source mnistm \
    --target svhn \
    --lr 1e-3 \
    --optim adam \
    --use_cuda \
    --gpu_num 0 \
    # --save \
    # --vis \
    # --eval \
    # --load best_model
