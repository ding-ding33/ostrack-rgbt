# Test baseline on LasHeR dataset
CUDA_VISIBLE_DEVICES=1 python ./RGBT_workspace/test_rgbt_mgpus.py --script_name bat --dataset_name LasHeR --yaml_name rgbt_baseline --epoch 40
