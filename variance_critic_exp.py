import copy
import glob
import itertools
import os
import pickle
import time
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

def main():

    mode = 'PPO'

    # Args basically
    use_gae = True
    gae_lambda = 0.95
    gamma = 0.99
    entropy_coef = 0
    value_loss_coef = 0.5
    num_mini_batch = 32
    ppo_epoch = 10
    use_proper_time_limits = True
    num_steps_list = [2000]
    num_baseline_steps = 2500
    save_interval = 100
    log_interval = 10
    num_processes = 20
    num_env_steps = 10000000
    use_linear_lr_decay = True
    clip_param = 0.2
    lr = 1e-4
    eps = 1e-8
    max_grad_norm = 0.5

    seed = 1234
    env_name = "Humanoid-v2"
    log_dir = '/tmp/true_grad/'

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    envs = make_vec_envs(env_name, seed, num_processes,
                         gamma, log_dir, device, True)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': False})
    actor_critic.to(device)

    if mode == 'PPO':
        agent = algo.PPO(
            actor_critic,
            clip_param,
            ppo_epoch=ppo_epoch,
            num_mini_batch=num_mini_batch,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            lr=lr,
            eps=eps,
            max_grad_norm=max_grad_norm)

    # elif mode == 'VPG':
    #     agent = algo.VPG(
    #         actor_critic_1,
    #         value_loss_coef=0.5,
    #         entropy_coef=0,
    #         lr=None,
    #         eps=eps,
    #         max_grad_norm=None)

    # elif mode == 'VPG-NoBaseline':
    #     no_baseline_agent = algo.VPG(
    #         actor_critic_2,
    #         value_loss_coef=0,
    #         entropy_coef=0,
    #         has_value=False,
    #         lr=None,
    #         eps=eps,
    #         max_grad_norm=None)
    # elif mode == 'TRPO':
    else:
        raise Exception('Algorithm not specified correctly')

    # rollouts = RolloutStorage(num_steps, num_processes,
    #                           envs.observation_space.shape, envs.action_space,
    #                           actor_critic.recurrent_hidden_state_size)

    # obs = envs.reset()
    # rollouts.obs[0].copy_(obs)
    # rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    # num_updates = int(
    #     num_env_steps) // num_steps // num_processes

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': False})
    actor_critic.to(device)

    agent = algo.PPO(
        actor_critic,
        clip_param,
        ppo_epoch=ppo_epoch,
        num_mini_batch=num_mini_batch,
        value_loss_coef=value_loss_coef,
        entropy_coef=entropy_coef,
        lr=lr,
        eps=eps,
        max_grad_norm=max_grad_norm)

    num_updates = 1
    ave_cos_sims = []

    num_steps_list_norm = torch.tensor(num_steps_list) / num_processes

    num_steps = num_steps_list_norm[0].item()
    num_steps_true = int(500000 / num_processes)

    rollouts = RolloutStorage(num_steps, num_processes,
                    envs.observation_space.shape, envs.action_space,
                    actor_critic.recurrent_hidden_state_size)

    rollouts_true = RolloutStorage(num_steps_true, num_processes,
                    envs.observation_space.shape, envs.action_space,
                    actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    eval_epoch = 150

    for j in range(eval_epoch):

        grad_estimates = []
        if j < eval_epoch - 1:
            for step in range(num_steps_true):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts_true.obs[step], rollouts_true.recurrent_hidden_states[step],
                        rollouts_true.masks[step])

                # Obser reward and next obs
                obs, reward, done, infos = envs.step(action)

                for info in infos:
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])

                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                    for info in infos])
                if step < num_steps:
                    rollouts.insert(obs, recurrent_hidden_states, action,
                                    action_log_prob, value, reward, masks, bad_masks)
                rollouts_true.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)

            with torch.no_grad():
                next_value_true = actor_critic.get_value(
                    rollouts_true.obs[-1], rollouts_true.recurrent_hidden_states[-1],
                    rollouts_true.masks[-1]).detach()
                next_value = actor_critic.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()
            rollouts_true.compute_returns(next_value_true, False, 1.00,
                                    gae_lambda, use_proper_time_limits)
            rollouts.compute_returns(next_value, use_gae, gamma,
                                    gae_lambda, use_proper_time_limits)
            value_loss, action_loss, dist_entropy = agent.update(rollouts_true, train_only_critic=True)
            value_loss, action_loss, dist_entropy = agent.update(rollouts, train_critic=False)
            rollouts.after_update()
            rollouts_true.after_update()
        else:
            for _ in range(10):
                for step in range(num_steps):
                    # Sample actions
                    with torch.no_grad():
                        value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                            rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                            rollouts.masks[step])

                    # Obser reward and next obs
                    obs, reward, done, infos = envs.step(action)

                    for info in infos:
                        if 'episode' in info.keys():
                            episode_rewards.append(info['episode']['r'])

                    # If done then clean the history of observations.
                    masks = torch.FloatTensor(
                        [[0.0] if done_ else [1.0] for done_ in done])
                    bad_masks = torch.FloatTensor(
                        [[0.0] if 'bad_transition' in info.keys() else [1.0]
                        for info in infos])
                    rollouts.insert(obs, recurrent_hidden_states, action,
                                    action_log_prob, value, reward, masks, bad_masks)

                with torch.no_grad():
                    next_value = actor_critic.get_value(
                        rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                        rollouts.masks[-1]).detach()

                rollouts.compute_returns(next_value, use_gae, gamma,
                                        gae_lambda, use_proper_time_limits)
                
                grad_estimates.append(agent.get_grad_vector(rollouts))

                # cosine_sim = torch.nn.functional.cosine_similarity(grad_full, grad_est, dim=0)
                # print("Cosine similarity:", cosine_sim.item())

                rollouts.after_update()
                rollouts_true.after_update()

            cos_sims = []
            for a, b in itertools.combinations(grad_estimates, 2):
                cos_sims.append(torch.nn.functional.cosine_similarity(a, b, dim=0))
            cos_sims = torch.tensor(cos_sims)
            ave_cos_sims.append(cos_sims.mean().item())

        # save for every interval-th episode or for the last epoch
        # if (j % save_interval == 0
        #         or j == num_updates - 1) and args.save_dir != "":
        #     save_path = os.path.join(args.save_dir, args.algo)
        #     try:
        #         os.makedirs(save_path)
        #     except OSError:
        #         pass

        #     torch.save([
        #         actor_critic,
        #         getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
        #     ], os.path.join(save_path, args.env_name + ".pt"))

        if j % log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * num_processes * num_steps
            end = time.time()

            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        # if (args.eval_interval is not None and len(episode_rewards) > 1
        #         and j % args.eval_interval == 0):
        #     ob_rms = utils.get_vec_normalize(envs).ob_rms
        #     evaluate(actor_critic, ob_rms, args.env_name, args.seed,
        #              args.num_processes, eval_log_dir, device)
    df = pd.DataFrame({'batch_sizes': num_steps_list, 'ave-cos-sim': ave_cos_sims})
    df.to_csv('./store/variance-critic-cos-sim.csv')

if __name__ == "__main__":
    main()