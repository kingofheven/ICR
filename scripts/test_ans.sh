exp_name="bs128_grad-acc4_rk4_seed1208"
cvd=0
num_gpus=1
dataset="csn_java" #csn_java, csn_python, b2f_small, b2f_medium, conala
split="test"
main_process_port=$((RANDOM % 5001 + 25000))
ckpt="$PWD/exps/$exp_name/iter2_tree/model_ckpt/dpr_biencoder.3"
num_prompts=-1
generate_embedding_batch_size=2048
inference_out="$PWD/results/gpt-neo-2.7b/inference_${dataset}_${split}.json"
python gen_test.py --fp "${inference_out}" --dataset $dataset --split $split \