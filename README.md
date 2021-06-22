# VAI: Unsupervised Visual Attention and Invariance for Reinforcement Learning.

by [Xudong Wang*](http://people.eecs.berkeley.edu/~xdwang/), [Long Lian*](https://github.com/TonyLianLong/) and [Stella X. Yu](http://www1.icsi.berkeley.edu/~stellayu/) at UC Berkeley / ICSI. (*: equal contribution)

<em>IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.</em>

<p align="center">
  <img src="https://raw.githubusercontent.com/TonyLianLong/VAI-ReinforcementLearning/main/introduction.png"  width="100%" >
</p>

For more information, please check: [Project Page](http://people.eecs.berkeley.edu/~xdwang/projects/VAI/) | [PDF](http://people.eecs.berkeley.edu/~xdwang/papers/CVPR2021_VAI.pdf) | [Preprint](https://arxiv.org/abs/2104.02921) | [BibTex](http://people.eecs.berkeley.edu/~xdwang/papers/VAI.txt)

The key message of this paper: Adapt not RL, but the vision!

This repo has the official PyTorch VAI implementation.

- [VAI: Unsupervised Visual Attention and Invariance for Reinforcement Learning.](#vai-unsupervised-visual-attention-and-invariance-for-reinforcement-learning)
  - [How to run the code?](#how-to-run-the-code)
  - [FAQ](#faq)
  - [Pretrained Models](#pretrained-models)
  - [Citation](#citation)
  - [Credits](#credits)

## How to run the code?
### Preparation
Before you start, you need to make sure that you have a valid mujoco [key](http://www.mujoco.org/) and your environment satisfies the requirements of [OpenAI gym](https://github.com/openai/gym). Requirements for RL can be installed in a similar way as [PAD](https://github.com/nicklashansen/policy-adaptation-during-deployment).
If you use conda, you could install with this file:
```
conda env create -f install.yaml
conda activate vai
```

If you prefer configuring on your own, make sure to install packages in install.yaml.

Then, install the following python packages:
```
# If you need DeepMind Control suite:
cd src/env/dm_control
pip install -e .

cd ../dmc2gym
pip install -e .

# If you also need DrawerWorld.
cd ../drawerworld
pip install -e .
```

The code requires a copy of Places dataset on `places365_standard` directory (download the dataset on their [website](http://places2.csail.mit.edu/download.html)).

Note: this project requires a torch >= 1.7.0 and CUDA >= 10.2, and official NVIDIA driver is needed instead of the open-source NVIDIA driver. If you encounter mujoco/gym issues, please contact [mujoco](http://www.mujoco.org/) or [OpenAI gym](https://gym.openai.com/). If you encounter DM Control or DrawerWorld issues, you can have a look at [DM Control](https://github.com/deepmind/dm_control/) repo and [MetaWorld](https://meta-world.github.io/) repo first and see whether people have similar questions. DM Control and MetaWorld actually use different mujoco python package, so running one does not imply the other is configured right. We suggest Ubuntu 18.04 since packages may not be pre-compiled on other OS (e.g. CentOS). This can save you a lot of time!

After installation, you can start collecting samples from the environment and start training adapters.

### Step 0: Extract data from replay buffer by performing a random run
```
CUDA_VISIBLE_DEVICES=0 EGL_DEVICE_ID=0 python3 src/train.py     --domain_name cartpole     --task_name balance     --action_repeat 8     --mode train         --num_shared_layers 4     --seed 0    --work_dir logs/your_work_dir  --update_rewards   --save_model --num_layers 4 --use_inv --export_replay_buffer --init_steps 5000
```

This command runs the training script and `--export_replay_buffer` asks it to export the replay buffer and quit before training starts. Since it quits before training starts, all data is collected with a random policy.

Feel free to adjust `init_steps` if you find the collected data not enough. We use `5000` across all environments.

### Step 1: Train keypoints 

Train keypoints by running: 
```
CUDA_VISIBLE_DEVICES=0 python src/train_stage1.py
```

Feel free to adjust the hyperparams in the file. On environments other than cartpole, `k` needs to be tuned to select as much foreground and little background as possible. However, not much tuning needs to performed when compared to Step 2, since training environments don't have complex textures and keypoints model is not used in test environment. 

### Step 2: Train attention module

Please modify the path for keypoints in `train_stage2.py` to load the keypoints that you just trained. See comments for tuning hyperparams for different environments. This is much harder to tune because it needs to adapt in test environments. I strongly encourage you to export the adapted observation in test environment with `train.py` (you can use `mode` to switch to test environment in `train.py`) to check the adapter before RL training since RL training takes so long and you probably want to verify the adapter before the real training. Train adapter by running:

```
CUDA_VISIBLE_DEVICES=0 python train_stage2.py
```

### Step 3: Train RL

Use the following command to train RL:

```
CUDA_VISIBLE_DEVICES=0 EGL_DEVICE_ID=0 python3 src/train.py     --domain_name cartpole     --task_name balance     --action_repeat 8     --mode train         --num_shared_layers 4     --seed 0    --work_dir logs/your_work_dir  --update_rewards   --save_model --num_layers 4 --use_inv --adapt_observation --adapter_checkpoint adapter_checkpoint --adapt_aug
```

Note:

For finger tasks, we set `num_shared_layers=8`, `num_layers=11`. For other tasks we set `num_shared_layers=4`, `num_layers=4`.
For finger tasks, we set `action_repeat=2`. For cartpole tasks, we set `action_repeat=8`. For other tasks, `action_repeat=4`.

For DrawerWorld close and open, we set `action_repeat=2`. For DeepMind Control tasks, we follow the settings in PAD.

Please set work_dir to the working directory to save checkpoints, and set adapter_checkpoint to the path of checkpoint. Adapter is required in training time.

### Inference
The `eval.py` is for running inference. 

To run inference on many seeds, you probably want to use a shell script:
```
#!/bin/bash

for i in {0..9}
do
   echo "Running:" $i
   python3 src/eval.py \
    --domain_name cartpole \
    --task_name balance \
    --action_repeat 8 \
    --mode color_hard \
    # To enable video mode for individual video:
    # --mode video$i \
    --num_shared_layers 4 \
    --num_layers 4 \
    --seed $i \
    --work_dir logs/cartpole_balance_mask \
    --pad_checkpoint 500k \
    --adapt_observation \
    --adapter_checkpoint runs/May31_02-05-06_dm_control.pth \
    --pad_num_episodes 10
done

```

This file supports PAD too.

## FAQ
See [FAQ](FAQ.md). Please read the FAQ if you encounter any problems. We also offer some tips, since it can be very difficult to debug.

## Pretrained Models
[Download](https://drive.google.com/drive/folders/1RF1NTcG7caKiy3-VPVWLMIjoP2MdkGvn?usp=sharing)

## Citation
```
@InProceedings{Wang_2021_CVPR,
    author    = {Wang, Xudong and Lian, Long and Yu, Stella X.},
    title     = {Unsupervised Visual Attention and Invariance for Reinforcement Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {6677-6687}
}
```

## Credits
We thank authors for the following projects that we reference:
1. [PAD](https://github.com/nicklashansen/policy-adaptation-during-deployment) which our work is based on.
2. [A repo with our baseline SAC implementation](https://github.com/denisyarats/pytorch_sac_ae)
3. [DeepMind Control](https://github.com/deepmind/dm_control) which we benchmark on.
4. [Metaworld](https://meta-world.github.io/) which Drawerworld is based on.
5. [Transporter Implementation](https://github.com/ethanluoyc/transporter-pytorch)
6. [PyTorch Morphology](https://github.com/lc82111/pytorch_morphological_dilation2d_erosion2d) for functions that perform visual modifications.
7. [SODA](https://github.com/nicklashansen/dmcontrol-generalization-benchmark) for the code which loads Places.
