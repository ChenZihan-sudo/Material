# Operation Manual

## Setup

### Environment setup

1. Install `Anaconda3`, here is the installation guide on Linux: [Installing Anaconda3 on Linux](https://docs.anaconda.com/free/anaconda/install/linux/)
2. Check your CUDA version, using `nvidia-smi` in your terminal.
3. Create a new conda environment:

```bash
conda create -n <Environment name> python==3.12
```

Rename `<Environment name>` to your favorite one.

4. Install Pytorch, you can install `Pytorch 2.3.0` or a later version. You can find a suitable one with your CUDA version in here: [Pytorch previous versions](https://pytorch.org/get-started/previous-versions/)

```bash
# Grab your command from here: https://pytorch.org/get-started/previous-versions/
conda install pytorch==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

5. Install Pytorch geometric and related packages

Find a suitable torch version with CUDA version in here: [Pytorch geometric wheel](https://pytorch-geometric.com/whl/). You can copy the link like this: https://pytorch-geometric.com/whl/torch-2.3.0+cu118.html.

Run commands as follows:

```bash
torch_geometric_whl=<The wheel link you copied>

pip install torch-geometric==2.5.3
pip install torch-scatter -f $torch_geometric_whl
pip install torch-sparse -f $torch_geometric_whl
```

5. Install additional pip packages

```bash
pip install ase==3.22.1 ray==2.34.0 ray[tune]==2.34.0 pymatgen==2024.8.9 matplotlib tqdm tensorboard
```

### Configurations setup

1. Pick a path to place the project `Material` folder as the **absolute** work directory path. (The path should be as short as possible for the purpose of experiment restoring on other machines, but do NOT place it under any root paths like `/*`)

2. Change the `Default.absolute_work_dir` in `Material/config.yml` to your **absolute** work directory path.

3. Change the `Tuning.resources` and `Tuning.max_concurrent_trials` in `Material/config.yml` according to your hardware CPUs and GPUs capability.

   You should check how many CPUs and GPUs in the hardware first.

   For instance, If you have 16 CPUs(threads) and 1 GPU, you want to run trials in parallel to speed up the progress. You should consider your hardware capability especially the GPU memory to contain one more trial (btop and nvtop is recommended for checking the CPUs and GPUs resource usage). 

   If you find that 2 concurrent trials is bearable on your machine, you can divide your resources into 2 parts, as follows:

   ```bash
   # max concurrent trials if paralleled, 
   # set this according to your task and hardware resources
   max_concurrent_trials: 2
   resources:
     num_cpus: 16 # total hardware cpu(thread) resources
     num_gpus: 1 # total hardware gpu resources
     trial_cpus: 8 # 16/2=8, a trail uses trial_cpus=num_cpus/concurrent_trials if paralleled
     trial_gpus: 0.5 # 1/2=0.5, a trial uses trial_gpus=num_gpus/concurrent_trials if paralleled
   ```

   If you find that 4 concurrent trials is bearable on your machine, you can divide your resources into 2 	parts, as follows:

   ```bash
   # max concurrent trials if paralleled, 
   # set this according to your task and hardware resources
   max_concurrent_trials: 4
   resources:
     num_cpus: 16 # total hardware cpu(thread) resources
     num_gpus: 1 # total hardware gpu resources
     trial_cpus: 4 # 16/4=4, a trail uses trial_cpus=num_cpus/concurrent_trials if paralleled
     trial_gpus: 0.25 # 1/4=0.25, a trial uses trial_gpus=num_gpus/concurrent_trials if paralleled
   ```


## Usage

This will run a tuning task on the system backstage. Log file nohup.out` will record all system out and system error logs, you can check it under `absolute_work_dir` path.

```bash
cd <Your Default.absolute_work_dir>
nohup python -u main.py -M="ChemGNN" -D="MPDataset" -T="Tuning" > nohup.out 2>&1 &
```

You can record the `PID` for further stopping the process and it will be shown on the next line, as in `[1] 11102`.

## Results analysis

```bash
python main.py -T="TuningAnalysis" --tune_exp_path=<Your tune experiment path>
```

The format of tune experiment folder name is like this: `<Model>_<Dataset>_<Year>-<Month>-<Day>_<Hour>-<Minute>-<Second>`  is the experiment name in default. You can find the experiment name from the path `tune_result/tune/` in default.

The `tune_exp_path` should be the absolute path, as in `/home/catcolia/Material/tune_results/tune/ChemGNN_MPDataset_2024-09-09_21-15-41`

The tuning analysis will generate a file `tune_analysis_data.pt`

## Q&A

### 1. How to stop the current experiment process?

You can find the experiment process `PID` by `ps`, `top`, `btop`, etc. 

```bash
kill -SIGINT <PID>
```

This will kill the experiment process with a signal `SIGINT` attached. The current trial status will not be saved if you send the kill instructions once more as the force exit command.

### 2. How to restore the experiment?

After stopping the experiment process, you can add the experiment path to the `Tuning.restore_experiment_from` in `config.yml` to restore the experiment process. You can fill it like this format: `"{{Tuning.storage_path}}/<Model>_<Dataset>_<Year>-<Month>-<Day>_<Hour>-<Minute>-<Second>"`, as in `"{{Tuning.storage_path}}/ChemGNN_MPDataset_2024-08-27_09-53-22"`. You can find the experiment name from the path `tune_result/tune/` in default.

Then, you can start the program just like before, as in the same command in the chapter "Usage".

