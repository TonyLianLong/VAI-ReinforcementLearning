import numpy as np
import torch
import os
from copy import deepcopy
from tqdm import tqdm
import utils
from logger import Logger
from video import VideoRecorder

from arguments import parse_args
from env.wrappers import make_pad_env
from agent.agent import make_agent
from utils import get_curl_pos_neg

import time

def new_replay_buffer(env, args, capacity):
	replay_buffer = utils.ReplayBuffer(
		obs_shape=env.observation_space.shape,
		action_shape=env.action_space.shape,
		capacity=capacity,
		batch_size=args.pad_batch_size
	)

	return replay_buffer

def evaluate(env, agent, args, video, adapt=False):
	"""Evaluate an agent, optionally adapt using PAD"""
	episode_rewards = []
	episode_pred_rewards = []

	assert not args.use_curl # replay buffer needs to be handled separately for CURL

	replay_buffer = new_replay_buffer(env, args, capacity=env._max_episode_steps * args.pad_num_episodes)
	ep_agent = deepcopy(agent) # make a new copy
	L = Logger(args.work_dir, use_tb=False)

	step = 0

	if args.force_real_reward_adapt:
		print("Agent uses real reward to adapt")
	else:
		print("Agent uses predicted reward to adapt")

	# print("Using predicted reward from agent")

	if args.domain_name == "metaworld":
		success_num = 0

	for i in tqdm(range(args.pad_num_episodes)):		
		video.init(enabled=True)

		obs = env.reset()
		done = False
		episode_reward = 0
		episode_pred_reward = 0
		episode_step = 0
		losses = []

		if args.moving_average_denoise:
			obs = np.vstack((obs, obs))

		if adapt or (not args.no_adapt):
			ep_agent = deepcopy(agent) # make a new copy
		ep_agent.train()

		while not done:
			# Take step
			with utils.eval_mode(ep_agent):
				if args.eval_sample_action:
					action = ep_agent.sample_action(obs)
				else:
					action = ep_agent.select_action(obs)

			next_obs, reward, done, info = env.step(action)
			if args.moving_average_denoise:
				un_denoised_next_obs = info["un_denoised_obs"] # We should not use other things in info for training
				next_obs = np.vstack((next_obs, un_denoised_next_obs))

			pred_reward = ep_agent.predict_reward(utils.random_crop(torch.Tensor(obs).cuda().unsqueeze(dim=0)), utils.random_crop(torch.Tensor(next_obs).cuda().unsqueeze(dim=0))).item()
			# print(reward, pred_reward)

			episode_reward += reward
			episode_pred_reward += pred_reward
			
			# Make self-supervised update if flag is true
			if adapt:
				if args.use_rot: # rotation prediction

					# Prepare batch of cropped observations
					batch_next_obs = utils.batch_from_obs(torch.Tensor(next_obs).cuda(), batch_size=args.pad_batch_size)
					batch_next_obs = utils.random_crop(batch_next_obs)

					# Adapt using rotation prediction
					losses.append(ep_agent.update_rot(batch_next_obs))
				
				if args.use_inv: # inverse dynamics model

					# Prepare batch of observations
					batch_obs = utils.batch_from_obs(torch.Tensor(obs).cuda(), batch_size=args.pad_batch_size)
					batch_next_obs = utils.batch_from_obs(torch.Tensor(next_obs).cuda(), batch_size=args.pad_batch_size)
					batch_action = torch.Tensor(action).cuda().unsqueeze(0).repeat(args.pad_batch_size, 1)

					# Adapt using inverse dynamics prediction
					losses.append(ep_agent.update_inv_and_rewards(utils.random_crop(batch_obs), utils.random_crop(batch_next_obs), batch_action))

			if step >= args.init_steps and (not args.no_adapt):
				ep_agent.update(replay_buffer, L, step)

			done = 1 if episode_step + 1 == env._max_episode_steps else float(done)
			if args.domain_name == "metaworld":
				if info["success"]:
					done = 1
					success_num += 1
			if not args.no_adapt:
				if args.force_real_reward_adapt:
					replay_buffer.add(obs, action, reward, next_obs, done)
				else:
					replay_buffer.add(obs, action, pred_reward, next_obs, done)

			video.record(env, losses)
			obs = next_obs
			step += 1
			episode_step += 1

		video.save(f'{args.mode}_pad_{i}.mp4' if adapt else f'{args.mode}_eval_{i}.mp4')

		# print("Episode reward:", episode_reward, ", episode pred reward:", episode_pred_reward)

		episode_rewards.append(episode_reward)
		episode_pred_rewards.append(episode_pred_reward)

	if args.eval_save_episode_rewards:
		np.save(args.eval_save_episode_rewards, [episode_rewards, episode_pred_rewards])
	
	np.save(os.path.join(args.work_dir, f'log_{args.mode}_episode_rewards_{int(time.time())}.npy'), episode_rewards)

	print("Mean episode pred reward: ", np.mean(episode_pred_rewards))
	if args.domain_name == "metaworld":
		print(f"Success rate: {success_num/args.pad_num_episodes * 100:1}%")
	return np.mean(episode_rewards)


def init_env(args):
		utils.set_seed_everywhere(args.seed)
		return make_pad_env(
			domain_name=args.domain_name,
			task_name=args.task_name,
			seed=args.seed,
			episode_length=args.episode_length,
			action_repeat=args.action_repeat,
			mode=args.mode,
			action_factor=args.action_factor,
			action_bias=args.action_bias,
			action_noise_factor=args.action_noise_factor,
			moving_average_denoise=args.moving_average_denoise,
			moving_average_denoise_factor=args.moving_average_denoise_factor,
			moving_average_denoise_alpha=args.moving_average_denoise_alpha,
			exponential_moving_average=args.exponential_moving_average
		)


def main(args):
	# Initialize environment
	env = init_env(args)
	model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
	video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
	video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)

	# Prepare agent
	assert torch.cuda.is_available(), 'must have cuda enabled'
	cropped_obs_shape = (3*args.frame_stack, 84, 84)
	agent = make_agent(
		obs_shape=cropped_obs_shape,
		action_shape=env.action_space.shape,
		args=args
	)
	agent.load(model_dir, args.pad_checkpoint, load_optimizers=not args.no_load_optimizers, load_target=args.load_target)

	

	# Evaluate agent with PAD (if applicable)
	pad_reward = None
	eval_reward = None
	
	if args.use_inv or args.use_curl or args.use_rot:
		env = init_env(args)
		print(f'Policy Adaptation during Deployment (PAD) of {args.work_dir} for {args.pad_num_episodes} episodes (mode: {args.mode})')
		pad_reward = evaluate(env, agent, args, video, adapt=True)
		print('Mean pad reward:', int(pad_reward))
	else:
		# Evaluate agent without PAD
		print(f'Evaluating {args.work_dir} for {args.pad_num_episodes} episodes (mode: {args.mode})')
		eval_reward = evaluate(env, agent, args, video)
		print('Mean eval reward:', int(eval_reward))

	# Save results
	results_fp = os.path.join(args.work_dir, f'pad_{args.mode}.pt')
	torch.save({
		'args': args,
		'eval_reward': eval_reward,
		'pad_reward': pad_reward
	}, results_fp)
	print('Saved results to', results_fp)


if __name__ == '__main__':
	args = parse_args()
	main(args)
