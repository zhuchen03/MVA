#!/usr/bin/env bash

function runexp {
debug=0

GPU=${1}
TASK=${2}
NUM_CLASSES=${3}
optimizer=${4}          # adam
LR=${5}                # Peak LR for polynomial LR scheduler.
wd=${6}
b2_min=${7}
b2_max=${8}
ioffset=${9}
eps=${10} #1e-6
TOTAL_NUM_UPDATES=${11}  # 10 epochs through RTE for bsz 16
WARMUP_UPDATES=${12}      # 6 percent of the number of updates
MAX_SENTENCES=${13}        # Batch size.
freq=${14}
seed=${15}
gamma=${16}
ROBERTA_PATH=roberta-pretrained/roberta.base/model.pt

if [ ${optimizer} = "madam" ]; then
betas="'(${b2_min},${b2_max})'"
opt_commands="--optimizer madam --index-offset ${ioffset} --beta1 0.9 --beta2-range ${betas} --adam-eps ${eps} "
expname="${TASK}-v2-rbase-madam-lr${LR}-wd${wd}-ioff${ioffset}-b2_${b2_min}_${b2_max}-eps${eps}-seed${seed}-quad3"

elif [ ${optimizer} = "lamadam" ]; then
opt_commands="--optimizer lamadam  --momentum 0.9 --beta-min ${b2_min} --varscale-beta ${b2_max} --varscale-eps ${eps} "
expname="${TASK}-rbase-lamadam-lr${LR}-wd${wd}-b2_${b2_min}_${b2_max}-eps${eps}-seed${seed}"

fi

if [ ${TASK} = "STS-B"  ]; then
other_commands="--best-checkpoint-metric pearson --maximize-best-checkpoint-metric --regression-target"
elif [ ${TASK} = "CoLA" ]; then
other_commands="--best-checkpoint-metric mcc --maximize-best-checkpoint-metric "
else
other_commands="--best-checkpoint-metric accuracy --maximize-best-checkpoint-metric "
fi

cmd="
CUDA_VISIBLE_DEVICES=${GPU} python train.py data-bin/glue/${TASK}-bin/ \
    --restore-file $ROBERTA_PATH \
    --save-dir chks-glue/${TASK}/${expname} \
    --max-positions 512 \
    --max-sentences $MAX_SENTENCES --update-freq ${freq} \
    --max-tokens 4400 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_base \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay ${wd} ${opt_commands} \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 10 \
    --keep-last-epochs 0 --log-interval 20 --no-save-optimizer-state \
    --find-unused-parameters \
    --seed ${seed} ${other_commands} \
"

if [ ${debug} -eq 0 ]; then
logpath="logs/${expname}.log"
cmd="${cmd}  --comet --comet-project 'GLUE-beta2-base-${TASK}' --comet-real-tag init --comet-tag ${expname}
> ${logpath} 2>&1 &
"
echo ${logpath}
fi

eval ${cmd}

}

runexp   0   SST-2     2      madam     2e-05      0.1      0.5    0.98      0      1e-6           20935            1256              32       1   4566
