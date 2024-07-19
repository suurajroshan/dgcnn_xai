MODEL='dgcnn'           # choices=['dgcnn']
DATA_PATH=''
SAVE_PATH=''

NUM_POINTS=2048
PC_ATTRIBS='xyzNxNyNz' # options: xyz, xyzNxNyNz, xyzNxNyNzRGB

EXP_NAME='semseg_s3dis_6'
TEST_AREA=6            # choices=['1', '2', '3', '4', '5', '6', 'all'])

BATCH_SIZE=8
EPOCHS=100
LR=0.001
MOMENTUM=0.9
SCHEDULER='step'       # choices=['step', 'cos']
NO_CUDA=False
DROPOUT=0.5
K=20

args=(--model "$MODEL" --data_path  "$DATA_PATH" 
--save_path "$SAVE_PATH" --num_points $NUM_POINTS --pc_attribs "$PC_ATTRIBS" 
--exp_name "$EXP_NAME" --test_area $TEST_AREA --batch_size $BATCH_SIZE 
--epochs $EPOCHS --lr $LR --momentum $MOMENTUM --scheduler $SCHEDULER 
--no_cuda $NO_CUDA --dropout $DROPOUT --k $K)

CUDA_VISIBLE_DEVICES=$GPU_ID python dgcnn_utils/main.py "${args[@]}"