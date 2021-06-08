# VAI: Unsupervised Visual Attention and Invariance for Reinforcement Learning.

by Xudong Wang*, Long Lian* and Stella X. Yu at UC Berkeley / ICSI. (*: equal contribution)

<em>IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.</em>

<p align="center">
  <img src="https://people.eecs.berkeley.edu/~xdwang/projects/VAI/VAI2.png"  width="100%" >
</p>

For more information, please check: [PDF](https://people.eecs.berkeley.edu/~xdwang/papers/CVPR2021VAI.pdf) | 
[Preprint](https://arxiv.org/abs/2104.02921) | [BibTex](https://people.eecs.berkeley.edu/~xdwang/papers/VAI.txt)

The key message of this paper: Adapt not RL, but the vision!

This repo has the official PyTorch VAI implementation.

- [VAI: Unsupervised Visual Attention and Invariance for Reinforcement Learning.](#vai-unsupervised-visual-attention-and-invariance-for-reinforcement-learning)
  - [How to run the code?](#how-to-run-the-code)
  - [FAQ](#faq)
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
### I got a mujoco error (e.g. C error/python error in installation or running), what should I do?
Since mujoco requires C part (and rendering), such error may occur if your system misses packages. Please double check your packages and contact mujoco for help. 

### I see multiple options in gym to render. What should I use to render?
As long as it comes with GPU acceleration, it should be fast. I use EGL since I'm running on a server without screen and you need to make sure EGL is supported if you use EGL. This depends on your NVIDIA driver and can be hard to debug.

### What are the `--update_rewards` and `--use_inv` in the example command?
These are legacy for compatability purpose. This codebase supports predicting the rewards and inverse dynamics as well as rotation as a side training task. These are useful for running policy adaptation during deployment (PAD experiments). This is enabled in training for compatability in loading our previously-trained checkpoints. This is not enabled in inference since it adds to latency and we don't need adaptation in the current setting. We experimented and we discovered no significant change when removing the flag of reward prediction.

### How do you benchmark video backgrounds?
We use the script above with `video0` for seed 0, `video1` for seed 1, etc. In the end, we take mean and std of all inference rewards sum.

### Hints for troubleshooting?
RL + vision is a task which is inherently very unstable and difficult to solve and even different runs could lead to different results if the algorithm is not stable enough. If you see something is wrong, debug in the following sequence: if the rewards in training environment do not meet the expectation, then it is probably *not* a problem of domain or visual changes, and in this case, you could try reducing augmentation and starting to tune the RL parameters (You may add, remove, or change RL augmentations or clamp in encoder.py). If the rewards in training environment do meet expectation, but the rewards in test environment do not, then you should check the performance of adapter in testing env. If you have no problem in color hard but problems in video tasks (as they are very challenging), you can export the frames before and after attention module and visualize (this is very helpful) in your attention training hyperparameters (you can do it before training RL and it is really helpful to use a good adapter, see `forward_conv` in encoder.py as an example).

In addition, pixels of background leakage often happen and it is normal to have it without influencing the performance. Also, since the training environments used by works in this series are with little distractions (e.g. not much texture in the background and easy to separate), the quality of masks will satisfy the requirement of the attention module most of the time (also, leaking patches of background is not as harmful as missing the foreground, so use a lower threshold). What is difficult is to adapt in video scenarios as they come with confusions in terms of texture and color, and tuning the hyperparameters may be required, although augmentation based on natural images already provides a pretty good augmentation when compared to hand-designed augmentations.

For hyperparam tuning, we have given a set of hyperparams that perform well enough in many scenarios, and so in Step 1, since the training environments do not have much noise and the foreground and background are relative distinguishable given their different color, the hyperparameters are relatively stable across environments. Including background patches do not influence much of the quality of the trained adapter module in stage 2, but missing information will cause issues in RL. You can use provided "dilate" function to make the mask larger, if you find the masks to miss information, although it's not used in our final training to get foreground in stage 2. In addition, for Step 2, a common thing is that foreground patches are missing in videos. This is mode severe than including background patches. Adjusting weights for foreground could help because the pixels for positive/negative samples are imbalanced.

For DeepMind Control, although Places dataset is not required to get improvements on the color hard environment, Places dataset is recommended for the video environment since it involves complicated textures which make the adapter hard to distinguish. For DrawerWorld, since the texture is repeating (relatively uniform), the adapter works even without Places dataset. If you decide to use Places augmentation, you need to control the strength of it to avoid mis-classify patches. In addition, in the experience of training adapter, the DrawerWorld benchmark is not difficult in training environment and the training environment has simple foreground and background, so hyperparam "k" in Stage 1 does not matter much for DrawerWorld and algorithm can select foreground efficiently even if you over-specify "k". The benchmark is designed to be difficult in the test environment and they are very challenging in terms of foreground and background separation because the texture has never appeared to the model in training time, so Stage 2 needs a lot of attention.

### Are you going to release Pretrained Models?
We are still sorting out the models, given that each model is hard to manage as it has multiple files. We plan to release pretrained models after it so that you could have a try of our models on different (possibly even harder) tasks.

## Credits
We thank authors for the following projects that we reference:
1. [PAD](https://github.com/nicklashansen/policy-adaptation-during-deployment) which our work is based on.
2. [A repo with our baseline SAC implementation](https://github.com/denisyarats/pytorch_sac_ae)
3. [DeepMind Control](https://github.com/deepmind/dm_control) which we benchmark on.
4. [Metaworld](https://meta-world.github.io/) which Drawerworld is based on.
5. [Transporter Implementation](https://github.com/ethanluoyc/transporter-pytorch)
6. [PyTorch Morphology](https://github.com/lc82111/pytorch_morphological_dilation2d_erosion2d) for functions that perform visual modifications.
7. [SODA](https://github.com/nicklashansen/dmcontrol-generalization-benchmark) for the code which loads Places.
