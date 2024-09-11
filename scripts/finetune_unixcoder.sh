# Usage: train_and_test.sh model_name_or_path dataset_path output_dir tag_prefix
if [ $# -ne 4 ]; then
    echo "Usage: train_and_test.sh model_name_or_path dataset_path output_dir tag_prefix"
    exit 1
fi

echo "Working Directory: $(pwd)"

# Arguments
ARG_model_name_or_path=$1
ARG_dataset_path=$2
ARG_output_dir=$3
ARG_tag_prefix=$4

# Config
max_source_length=1000
max_target_length=24
beam_size=10
train_batch_size=48
eval_batch_size=48
learning_rate=5e-5
gradient_accumulation_steps=2
num_train_epochs=10

# Python: assert max_source_length + max_target_length <= 1024
if [ $((max_source_length + max_target_length)) -gt 1024 ]; then
	echo "max_source_length + max_target_length must be less than or equal to 1024."
	exit 1
fi

# Python: model_name_or_path_short = os.path.basename(model_name_or_path).replace('-', '_')
model_name_or_path_short=$(basename ${ARG_model_name_or_path} | tr '-' '_')
echo "model_name_or_path_short: ${model_name_or_path_short}"

# Python: dataset_path_short = os.path.basename(dataset_path).split('.')[1]
dataset_path_short=$(basename ${ARG_dataset_path} | cut -d'.' -f2)
echo "dataset_path_short: ${dataset_path_short}"

# Generate `tag` by the below args and current datetime
## Example: tag="model1-BMunixcoder_base-DSmin-msl256-mtl128-beamS10-tbs2-ebs2-lr5e-5-gas2-nte10"
tag="${ARG_tag_prefix}"
tag="${tag}-BM${model_name_or_path_short}"
tag="${tag}-DS${dataset_path_short}"
tag="${tag}-msl${max_source_length}"
tag="${tag}-mtl${max_target_length}"
tag="${tag}-beamS${beam_size}"
tag="${tag}-tbs${train_batch_size}"
tag="${tag}-ebs${eval_batch_size}"
tag="${tag}-lr${learning_rate}"
tag="${tag}-gas${gradient_accumulation_steps}"
tag="${tag}-nte${num_train_epochs}"
tag="${tag}-date$(date +%Y%m%d%H%M)"
echo "Tag: ${tag}"

model_name_or_path="${ARG_model_name_or_path}"
dataset_path="${ARG_dataset_path}"
output_dir="${ARG_output_dir}/${tag}"
echo "model_name_or_path: ${model_name_or_path}"
echo "dataset_path: ${dataset_path}"
echo "output_dir: ${output_dir}"
echo "See Log: ${output_dir}/train-*.log"
mkdir -p ${output_dir}

# Training
echo -e "\033[34m================= Training =================\033[0m"
python -u -W ignore scripts/finetune_unixcoder.py \
	--do_train \
	--do_eval \
	--model_name_or_path ${model_name_or_path} \
	--train_filename "${dataset_path}/train.jsonl" \
	--dev_filename "${dataset_path}/valid.jsonl" \
	--output_dir ${output_dir} \
	--max_source_length ${max_source_length} \
	--max_target_length ${max_target_length} \
	--beam_size ${beam_size} \
	--train_batch_size ${train_batch_size} \
	--eval_batch_size ${eval_batch_size} \
	--learning_rate ${learning_rate} \
	--gradient_accumulation_steps ${gradient_accumulation_steps} \
	--num_train_epochs ${num_train_epochs}

if [ $? -ne 0 ]; then
    echo "Training failed."
    exit 1
fi

# Evaluating
echo -e "\033[34m================= Evaluating =================\033[0m"
python -u -W ignore scripts/finetune_unixcoder.py \
	--do_test \
	--model_name_or_path ${model_name_or_path} \
	--test_filename "${dataset_path}/test.jsonl" \
	--output_dir ${output_dir} \
	--max_source_length ${max_source_length} \
	--max_target_length ${max_target_length} \
	--beam_size ${beam_size} \
	--train_batch_size ${train_batch_size} \
	--eval_batch_size ${eval_batch_size} \
	--learning_rate ${learning_rate} \
	--gradient_accumulation_steps ${gradient_accumulation_steps} \
	--num_train_epochs ${num_train_epochs}

if [ $? -ne 0 ]; then
    echo "Evaluating failed."
    exit 1
fi

# Do Evaluation (BLEU, Accuracy)
## TODO: ...
