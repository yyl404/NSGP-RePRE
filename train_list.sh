your_devices=1

CFGS=(
    cl_faster_rcnn_cfgs/incremental_task/cl_faster_rcnn_nsgp_repre_15_5_1.py
    cl_faster_rcnn_cfgs/incremental_task/cl_faster_rcnn_nsgp_repre_15_5_2.py
)


for CFG in "${CFGS[@]}"
do 
    echo $CFG
    # CUDA_VISIBLE_DEVICES=${your_devices} bash tools/dist_train.sh $CFG 2
    CUDA_VISIBLE_DEVICES=${your_devices} python tools/train.py $CFG
done
