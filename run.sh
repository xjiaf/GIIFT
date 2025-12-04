#!/bin/bash

gpu_id=0  # Single-GPU: specify GPU id

# ===== Dataset Selection =====
# Set to "multi30k" or "wmt"
dataset="multi30k"
# For WMT, set: dataset="wmt"

# ===== Execution Options =====
run_stage1=true   # Set to true to run Stage 1 (Captioning)
run_stage2=true   # Set to true to run Stage 2 (Translation)

# ===== Backbone Options =====
backbones=("mbart")
# If you only want MBART, set: backbones=("mbart")

# ===== Training Continuation =====
continue_training_stage1=false
continue_training_stage2=false

# ===== Parameter Lists =====
if [ "${backbones[0]}" == "mbart" ]; then
  learning_rates_stage1=(2e-5)  # Learning rate for Stage 1
  learning_rates_stage2=(1e-5)  # Learning rate for Stage 2
  num_layers_list=(9)
else
  learning_rates_stage1=(1e-4)  # LR for Stage 1
  learning_rates_stage2=(5e-5)  # LR for Stage 2
  num_layers_list=(13)
fi

mask_probs_stage1=(0)         # Mask probability for Stage 1
epochs_list_stage1=(50)       # Training epochs for Stage 1 (Captioning)
epochs_stage2=50              # Training epochs for Stage 2 (Translation)
prefix_lengths=(1)
batch_sizes=(64)
num_heads_list=(8)
mapping_networks=("gatl")     # e.g., "gine", "mix", etc.
languages=("de" "fr")         # e.g., "de", "fr", etc.

# Additional parameter (only used in WMT Stage 1 script; can be empty)
layerwise_unfreezing_args_stage1=""

# Whether to use English as the caption language in Stage 1
use_src_caption_stage1=false

# Whether to use --use_mbart_encoder and --use_fusion in each stage
use_mbart_encoder_stage1=true
use_mbart_encoder_stage2=true
use_fusion_stage1=true
use_fusion_stage2=true

# Whether to use --use_gate in each stage
use_gate_stage1=true
use_gate_stage2=true

use_subgraphs=false  # not used
subgraph_ratios=(0 0.25 0.5 0.75 1)

# ===== Other Fixed Parameters =====
num_gpus=1
update_count=4

# ===== Dataset-specific Settings =====
if [ "$dataset" == "wmt" ]; then
  stage1_mn_prefix="wmt"
  stage1_ds_arg=""
  stage1_test_ds="2016 val"
  stage2_mn_prefix="wmt"
  stage2_ds_arg="--ds wmt"
  stage2_test_ds="2014 val"
  test_datasets=("2014:wmt")
else
  stage1_mn_prefix="multi30k"
  stage1_ds_arg=""
  stage1_test_ds="2016 val"
  stage2_mn_prefix="multi30k"
  stage2_ds_arg=""
  stage2_test_ds="2016 val"
  test_datasets=("2016:flickr" "2017:flickr" "2017:mscoco")
fi


# ===== Main Loops =====
if [ "$use_subgraphs" = true ]; then
  for subgraph_ratio in "${subgraph_ratios[@]}"; do
    echo "==== Running experiments with subgraph_ratio = $subgraph_ratio ===="
    for backbone in "${backbones[@]}"; do
      for prefix_length in "${prefix_lengths[@]}"; do
        for bs in "${batch_sizes[@]}"; do
          for lr1 in "${learning_rates_stage1[@]}"; do
            for lr2 in "${learning_rates_stage2[@]}"; do
              for num_heads in "${num_heads_list[@]}"; do
                for num_layers in "${num_layers_list[@]}"; do
                  for mapping_network in "${mapping_networks[@]}"; do
                    for lang in "${languages[@]}"; do
                      for epochs_stage1 in "${epochs_list_stage1[@]}"; do
                        for mask_prob_stage1 in "${mask_probs_stage1[@]}"; do

                          # Handle use_src_caption parameter
                          if [ "$use_src_caption_stage1" = true ]; then
                            use_src_caption_arg_stage1="--use_src_caption"
                          else
                            use_src_caption_arg_stage1=""
                          fi

                          # Handle use_gate for Stage 1 and Stage 2
                          if [ "$use_gate_stage1" = true ]; then
                            use_gate_arg_stage1="--use_gate"
                          else
                            use_gate_arg_stage1=""
                          fi
                          if [ "$use_gate_stage2" = true ]; then
                            use_gate_arg_stage2="--use_gate"
                          else
                            use_gate_arg_stage2=""
                          fi

                          # Handle use_mbart_encoder for Stage 1 and Stage 2
                          if [ "$use_mbart_encoder_stage1" = true ]; then
                            use_mbart_encoder_arg_stage1="--use_mbart_encoder"
                          else
                            use_mbart_encoder_arg_stage1=""
                          fi
                          if [ "$use_mbart_encoder_stage2" = true ]; then
                            use_mbart_encoder_arg_stage2="--use_mbart_encoder"
                          else
                            use_mbart_encoder_arg_stage2=""
                          fi

                          # Handle use_fusion for Stage 1 and Stage 2
                          if [ "$use_fusion_stage1" = true ]; then
                            use_fusion_arg_stage1="--use_fusion"
                          else
                            use_fusion_arg_stage1=""
                          fi
                          if [ "$use_fusion_stage2" = true ]; then
                            use_fusion_arg_stage2="--use_fusion"
                          else
                            use_fusion_arg_stage2=""
                          fi

                          # Epoch arguments
                          epochs_arg_stage1="--epochs $epochs_stage1"
                          epochs_arg_stage2="--epochs $epochs_stage2"

                          # --ct arguments
                          if [ "$continue_training_stage1" = true ]; then
                            ct_arg_stage1="--ct"
                          else
                            ct_arg_stage1=""
                          fi
                          if [ "$continue_training_stage2" = true ]; then
                            ct_arg_stage2="--ct"
                          else
                            ct_arg_stage2=""
                          fi

                          # ---- Stage 1: ----
                          if [ "$run_stage1" = true ]; then
                            base_cmd_stage1="python src/main.py --num_gpus $num_gpus \
                              --mn ${stage1_mn_prefix}_${backbone} \
                              $stage1_ds_arg \
                              --prefix_length $prefix_length \
                              --bs $bs \
                              --update_count $update_count \
                              --lr $lr1 \
                              $epochs_arg_stage1 \
                              --test_ds $stage1_test_ds \
                              --stage caption \
                              --tgt_lang $lang \
                              --num_heads $num_heads \
                              --num_layers $num_layers \
                              --mapping_network $mapping_network \
                              --mask_prob $mask_prob_stage1 \
                              --backbone $backbone \
                              $use_src_caption_arg_stage1 $use_gate_arg_stage1 \
                              $use_mbart_encoder_arg_stage1 $use_fusion_arg_stage1 \
                              $layerwise_unfreezing_args_stage1 $ct_arg_stage1 \
                              --gpu_id $gpu_id \
                              --use_subgraphs --subgraph_ratio $subgraph_ratio"

                            echo "Running Stage 1 (Captioning) with backbone=$backbone, subgraph_ratio=$subgraph_ratio: $base_cmd_stage1"
                            eval $base_cmd_stage1
                            wait

                          else
                            echo "Skipping Stage 1 (Captioning) for backbone=$backbone"
                          fi

                          # ---- Stage 2: ----
                          if [ "$run_stage2" = true ]; then
                            if [ "$dataset" == "wmt" ]; then
                              bs_stage2=8
                            else
                              bs_stage2=$bs
                            fi

                            cmd_stage2="python src/main.py --num_gpus $num_gpus \
                              --mn ${stage2_mn_prefix}_${backbone} \
                              $stage2_ds_arg \
                              --prefix_length $prefix_length \
                              --bs $bs_stage2 \
                              --update_count $update_count \
                              --lr $lr2 \
                              $epochs_arg_stage2 \
                              --test_ds $stage2_test_ds \
                              --stage translate \
                              --tgt_lang $lang \
                              --lm model_pretrained.pth \
                              --num_heads $num_heads \
                              --num_layers $num_layers \
                              --mapping_network $mapping_network \
                              --backbone $backbone \
                              $use_gate_arg_stage2 $use_mbart_encoder_arg_stage2 $use_fusion_arg_stage2 $ct_arg_stage2 \
                              --gpu_id $gpu_id"
                            echo "Running Stage 2 (Translation) with backbone=$backbone: $cmd_stage2"
                            eval $cmd_stage2
                            wait

                          else
                            echo "Skipping Stage 2 (Translation) for backbone=$backbone"
                          fi

                          # ---- Testing Phase ----
                          for dataset_item in "${test_datasets[@]}"; do
                            IFS=":" read -r test_year test_mode <<< "$dataset_item"

                            if [ "$dataset" == "wmt" ]; then
                              bs_test=8
                            else
                              bs_test=$bs
                            fi

                            cmd_test="python src/main.py --num_gpus $num_gpus \
                              --mn ${stage2_mn_prefix}_${backbone} \
                              $stage2_ds_arg \
                              --src_lang en \
                              --tgt_lang $lang \
                              --prefix_length $prefix_length \
                              --bs $bs_test \
                              --test_ds $test_year $test_mode \
                              --stage translate \
                              --test \
                              --lm model_best_test.pth \
                              --num_heads $num_heads \
                              --num_layers $num_layers \
                              --mapping_network $mapping_network \
                              --backbone $backbone \
                              $use_gate_arg_stage2 $use_mbart_encoder_arg_stage2 $use_fusion_arg_stage2 \
                              --gpu_id $gpu_id"
                            echo "Testing with backbone=$backbone, year=$test_year, mode=$test_mode: $cmd_test"
                            eval $cmd_test
                            wait
                          done

                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
else

  for backbone in "${backbones[@]}"; do
    for prefix_length in "${prefix_lengths[@]}"; do
      for bs in "${batch_sizes[@]}"; do
        for lr1 in "${learning_rates_stage1[@]}"; do
          for lr2 in "${learning_rates_stage2[@]}"; do
            for num_heads in "${num_heads_list[@]}"; do
              for num_layers in "${num_layers_list[@]}"; do
                for mapping_network in "${mapping_networks[@]}"; do
                  for lang in "${languages[@]}"; do
                    for epochs_stage1 in "${epochs_list_stage1[@]}"; do
                      for mask_prob_stage1 in "${mask_probs_stage1[@]}"; do


                        # Handle use_src_caption parameter
                        if [ "$use_src_caption_stage1" = true ]; then
                          use_src_caption_arg_stage1="--use_src_caption"
                        else
                          use_src_caption_arg_stage1=""
                        fi

                        # Handle use_gate for Stage 1 and Stage 2
                        if [ "$use_gate_stage1" = true ]; then
                          use_gate_arg_stage1="--use_gate"
                        else
                          use_gate_arg_stage1=""
                        fi
                        if [ "$use_gate_stage2" = true ]; then
                          use_gate_arg_stage2="--use_gate"
                        else
                          use_gate_arg_stage2=""
                        fi

                        # Handle use_mbart_encoder for Stage 1 and Stage 2
                        if [ "$use_mbart_encoder_stage1" = true ]; then
                          use_mbart_encoder_arg_stage1="--use_mbart_encoder"
                        else
                          use_mbart_encoder_arg_stage1=""
                        fi
                        if [ "$use_mbart_encoder_stage2" = true ]; then
                          use_mbart_encoder_arg_stage2="--use_mbart_encoder"
                        else
                          use_mbart_encoder_arg_stage2=""
                        fi

                        # Handle use_fusion for Stage 1 and Stage 2
                        if [ "$use_fusion_stage1" = true ]; then
                          use_fusion_arg_stage1="--use_fusion"
                        else
                          use_fusion_arg_stage1=""
                        fi
                        if [ "$use_fusion_stage2" = true ]; then
                          use_fusion_arg_stage2="--use_fusion"
                        else
                          use_fusion_arg_stage2=""
                        fi

                        # Epoch arguments
                        epochs_arg_stage1="--epochs $epochs_stage1"
                        epochs_arg_stage2="--epochs $epochs_stage2"

                        # --ct arguments
                        if [ "$continue_training_stage1" = true ]; then
                          ct_arg_stage1="--ct"
                        else
                          ct_arg_stage1=""
                        fi
                        if [ "$continue_training_stage2" = true ]; then
                          ct_arg_stage2="--ct"
                        else
                          ct_arg_stage2=""
                        fi

                        # ---- Stage 1: ----
                        if [ "$run_stage1" = true ]; then
                          base_cmd_stage1="python src/main.py --num_gpus $num_gpus \
                            --mn ${stage1_mn_prefix}_${backbone} \
                            $stage1_ds_arg \
                            --prefix_length $prefix_length \
                            --bs $bs \
                            --update_count $update_count \
                            --lr $lr1 \
                            $epochs_arg_stage1 \
                            --test_ds $stage1_test_ds \
                            --stage caption \
                            --tgt_lang $lang \
                            --num_heads $num_heads \
                            --num_layers $num_layers \
                            --mapping_network $mapping_network \
                            --mask_prob $mask_prob_stage1 \
                            --backbone $backbone \
                            $use_src_caption_arg_stage1 $use_gate_arg_stage1 \
                            $use_mbart_encoder_arg_stage1 $use_fusion_arg_stage1 \
                            $layerwise_unfreezing_args_stage1 $ct_arg_stage1 \
                            --gpu_id $gpu_id"
                          echo "Running Stage 1 (Captioning) with backbone=$backbone: $base_cmd_stage1"
                          eval $base_cmd_stage1
                          wait

                        else
                          echo "Skipping Stage 1 (Captioning) for backbone=$backbone"
                        fi

                        # ---- Stage 2: ----
                        if [ "$run_stage2" = true ]; then
                          if [ "$dataset" == "wmt" ]; then
                            bs_stage2=8
                          else
                            bs_stage2=$bs
                          fi

                          cmd_stage2="python src/main.py --num_gpus $num_gpus \
                            --mn ${stage2_mn_prefix}_${backbone} \
                            $stage2_ds_arg \
                            --prefix_length $prefix_length \
                            --bs $bs_stage2 \
                            --update_count $update_count \
                            --lr $lr2 \
                            $epochs_arg_stage2 \
                            --test_ds $stage2_test_ds \
                            --stage translate \
                            --tgt_lang $lang \
                            --lm model_pretrained.pth \
                            --num_heads $num_heads \
                            --num_layers $num_layers \
                            --mapping_network $mapping_network \
                            --backbone $backbone \
                            $use_gate_arg_stage2 $use_mbart_encoder_arg_stage2 $use_fusion_arg_stage2 $ct_arg_stage2 \
                            --gpu_id $gpu_id"
                          echo "Running Stage 2 (Translation) with backbone=$backbone: $cmd_stage2"
                          eval $cmd_stage2
                          wait
                        else
                          echo "Skipping Stage 2 (Translation) for backbone=$backbone"
                        fi

                        # ---- Testing Phase ----
                        for dataset_item in "${test_datasets[@]}"; do
                          IFS=":" read -r test_year test_mode <<< "$dataset_item"

                          if [ "$dataset" == "wmt" ]; then
                            bs_test=8
                          else
                            bs_test=$bs
                          fi

                          cmd_test="python src/main.py --num_gpus $num_gpus \
                            --mn ${stage2_mn_prefix}_${backbone} \
                            $stage2_ds_arg \
                            --src_lang en \
                            --tgt_lang $lang \
                            --prefix_length $prefix_length \
                            --bs $bs_test \
                            --test_ds $test_year $test_mode \
                            --stage translate \
                            --test \
                            --lm model_best_test.pth \
                            --num_heads $num_heads \
                            --num_layers $num_layers \
                            --mapping_network $mapping_network \
                            --backbone $backbone \
                            $use_gate_arg_stage2 $use_mbart_encoder_arg_stage2 $use_fusion_arg_stage2 \
                            --gpu_id $gpu_id"
                          echo "Testing with backbone=$backbone, year=$test_year, mode=$test_mode: $cmd_test"
                          eval $cmd_test
                          wait
                        done

                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
fi
