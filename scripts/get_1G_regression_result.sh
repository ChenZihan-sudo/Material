#!/bin/bash
inner_model_path=$1
batch_size=$2
python main.py -T="Analysis" { -T="analyse_model_prediction" -D="OptimizedHypoDataset" -M="/home/catcolia/Material/results/ChemGNN/${inner_model_path}" -G="1G_sample" -B=${batch_size} }
python main.py -T="Analysis" { -T="regression_analysis" -D="OptimizedHypoDataset" -M="/home/catcolia/Material/results/ChemGNN/${inner_model_path}" -G="1G_sample" -B=${batch_size} }


