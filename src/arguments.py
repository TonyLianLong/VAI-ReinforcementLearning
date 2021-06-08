import argparse
import numpy as np
import warnings

def parse_args():
	parser = argparse.ArgumentParser()

	# environment
	parser.add_argument('--domain_name', default='walker')
	parser.add_argument('--task_name', default='walk')
	parser.add_argument('--frame_stack', default=3, type=int)
	parser.add_argument('--action_repeat', default=4, type=int)
	parser.add_argument('--episode_length', default=1000, type=int)
	parser.add_argument('--mode', default='train', type=str)
	
	# agent
	parser.add_argument('--init_steps', default=1000, type=int)
	parser.add_argument('--train_steps', default=500000, type=int)
	parser.add_argument('--batch_size', default=128, type=int)
	parser.add_argument('--hidden_dim', default=1024, type=int)

	# eval
	parser.add_argument('--save_freq', default=100000, type=int)
	parser.add_argument('--eval_freq', default=100000, type=int)
	parser.add_argument('--eval_episodes', default=10, type=int)

	# critic
	parser.add_argument('--critic_lr', default=1e-3, type=float)
	parser.add_argument('--critic_beta', default=0.9, type=float)
	parser.add_argument('--critic_tau', default=0.01, type=float)
	parser.add_argument('--critic_target_update_freq', default=2, type=int)

	# actor
	parser.add_argument('--actor_lr', default=1e-3, type=float)
	parser.add_argument('--actor_beta', default=0.9, type=float)
	parser.add_argument('--actor_log_std_min', default=-10, type=float)
	parser.add_argument('--actor_log_std_max', default=2, type=float)
	parser.add_argument('--actor_update_freq', default=2, type=int)

	# encoder
	parser.add_argument('--encoder_feature_dim', default=100, type=int)
	parser.add_argument('--encoder_lr', default=1e-3, type=float)
	parser.add_argument('--encoder_tau', default=0.05, type=float)

	# self-supervision
	parser.add_argument('--use_rot', default=False, action='store_true') # rotation prediction
	parser.add_argument('--use_inv', default=False, action='store_true') # inverse dynamics model
	parser.add_argument('--use_curl', default=False, action='store_true') # CURL
	parser.add_argument('--ss_lr', default=1e-3, type=float) # self-supervised learning rate
	parser.add_argument('--ss_update_freq', default=2, type=int) # self-supervised update frequency
	parser.add_argument('--num_layers', default=11, type=int) # number of conv layers
	parser.add_argument('--num_shared_layers', default=-1, type=int) # number of shared conv layers
	parser.add_argument('--num_filters', default=32, type=int) # number of filters in conv
	parser.add_argument('--curl_latent_dim', default=128, type=int) # latent dimension for curl
	
	# fitting rewards
	parser.add_argument('--update_rewards', default=False, action='store_true')
	parser.add_argument('--reward_lr', default=1e-3, type=float) # learning rate for reward fitting

	# sac
	parser.add_argument('--discount', default=0.99, type=float)
	parser.add_argument('--init_temperature', default=0.1, type=float)
	parser.add_argument('--alpha_lr', default=1e-4, type=float)
	parser.add_argument('--alpha_beta', default=0.5, type=float)

	# misc
	parser.add_argument('--seed', default=None, type=int)
	parser.add_argument('--work_dir', default=None, type=str)
	parser.add_argument('--save_model', default=False, action='store_true')
	parser.add_argument('--save_video', default=False, action='store_true')

	parser.add_argument('--fine_tune', default=False, action='store_true')
	parser.add_argument('--action_factor', default=1.0, type=float)
	parser.add_argument('--action_bias', default=0.0, type=float)
	parser.add_argument('--action_noise_factor', default=0.0, type=float)

	parser.add_argument('--moving_average_denoise', default=False, action='store_true')
	parser.add_argument('--moving_average_denoise_factor', default=0.15, type=float)
	parser.add_argument('--moving_average_denoise_alpha', default=0.2, type=float)
	parser.add_argument('--exponential_moving_average', default=0.0, type=float)
	
	# adapter
	parser.add_argument('--adapt_observation', default=False, action='store_true')
	parser.add_argument('--adapter_checkpoint', default=None, type=str)
	parser.add_argument('--adapt_aug', default=False, action='store_true')

	# test
	parser.add_argument('--eval_use_pred_rewards', default=False, action='store_true') # use pred rewards instead of true rewards
	parser.add_argument('--pad_checkpoint', default=None, type=str)
	parser.add_argument('--pad_batch_size', default=32, type=int)
	parser.add_argument('--pad_num_episodes', default=100, type=int)

	parser.add_argument('--no_load_optimizers', default=False, action='store_true')
	parser.add_argument('--load_target', default=False, action='store_true')
	parser.add_argument('--no_adapt', default=False, action='store_true')
	parser.add_argument('--eval_sample_action', default=False, action='store_true')
	parser.add_argument('--force_real_reward_adapt', default=False, action='store_true') # Only for debugging purpose
	parser.add_argument('--eval_save_episode_rewards', default=None, type=str, help="The filename to save episode rewards (npy).")

	parser.add_argument('--export_replay_buffer', default=False, action='store_true')

	args = parser.parse_args()

	if args.domain_name != "metaworld":
		assert args.mode in {'train', 'color_easy', 'color_hard'} or 'video' in args.mode, f'unrecognized mode "{args.mode}"'
	assert args.seed is not None, 'must provide seed for experiment'
	assert args.work_dir is not None, 'must provide a working directory for experiment'

	assert np.sum([args.use_inv, args.use_rot, args.use_curl]) <= 1, \
		'can use at most one self-supervised task'

	if args.adapter_checkpoint is None:
		args.adapter_checkpoint = f"adapter_checkpoints/{args.domain_name}_{args.task_name}.pth"

	if args.pad_checkpoint is not None:
		try:
			args.pad_checkpoint = args.pad_checkpoint.replace('k', '000')
			args.pad_checkpoint = int(args.pad_checkpoint)
		except:
			warnings.warn('pad_checkpoint is not int, received: %s' % args.pad_checkpoint)	
	return args
