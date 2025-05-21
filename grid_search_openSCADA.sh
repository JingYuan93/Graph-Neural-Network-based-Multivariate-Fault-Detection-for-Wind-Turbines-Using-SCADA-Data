CUDA_VISIBLE_DEVICES=0
DATASET="openSCADA"

seed=2024
BATCH_SIZE=32
SLIDE_WIN=6
SLIDE_STRIDE=1
val_ratio=0.3
decay=0
lr=0.0005

path_pattern="${DATASET}"
COMMENT="${DATASET}"

if [ ! -d "./logs/${path_pattern}" ]; then
    mkdir ./logs/${path_pattern}
fi

EPOCH=50
loss_type="recon"
report='val'
exp="openSCADA-网格搜索"

for dim in 64 128; do
    for topk in 3 5 7; do
        for out_layer_num in 1 2 3; do
            python main.py \
                -dataset $DATASET \
                -save_path_pattern $path_pattern \
                -slide_stride $SLIDE_STRIDE \
                -slide_win $SLIDE_WIN \
                -lr ${lr} \
                -batch $BATCH_SIZE \
                -epoch $EPOCH \
                -comment $COMMENT \
                -random_seed $seed \
                -dim $dim \
                -out_layer_num $out_layer_num \
                -out_layer_inter_dim $dim \
                -decay $decay \
                -val_ratio $val_ratio \
                -report $report \
                -topk $topk \
                -loss_type $loss_type \
                -exp $exp | tee logs/${path_pattern}/sw${SLIDE_WIN}_lr${lr}_ss${SLIDE_STRIDE}_bs${BATCH_SIZE}_ed${dim}_oln${out_layer_num}_old${dim}_vr${val_ratio}_report_${report}_topk${topk}_lt${loss_type}_${exp}.log
        done
    done
done