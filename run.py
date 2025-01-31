import copy
import glob
import os
import sys
import time
from collections import deque
import logging

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

from baselines.ppo2.ppo2 import safemean
from baselines import logger

from arguments import get_args
from representation import SimCLR
from envs import make_vec_envs

logging.basicConfig(level=logging.INFO)
import pdb

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    writer = SummaryWriter(comment=args.log_dir)
    log_dir = os.path.join(writer.log_dir, 'logging')
    save_dir = os.path.join(writer.log_dir, 'policy_checkpoints')
    logger.configure(log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    device = 'cuda' if args.cuda else "cpu"
    #For ProcGen could remove baselines dependency (but still need wrappers)
    #venv = ProcgenEnv(num_envs=args.num_processes, env_name=args.env_name)
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, writer.log_dir, device, False)
    print("No Framestack at the momement")
    actor_critic = Policy(
        (args.encoding_size,),
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})

    actor_critic.to(device)

    # Representation Learning Module
    simclr = SimCLR(envs.observation_space.shape, writer, device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        raise Exception('Gail needs to work with encoding')
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))

        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              (args.encoding_size,), envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    obs = simclr.encode(obs)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    epinfo = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            simclr.insert(obs)
            obs = simclr.encode(obs)

            for info in infos:
                if 'episode' in info.keys():
                    epinfo.append(info['episode'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])

            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        #Update Encodings
        obs = simclr.update_encoder()
        with torch.no_grad():
            for step, ob in enumerate(obs):
                value, _, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
                rollouts.insert(ob, recurrent_hidden_states, rollouts.actions[step],
                           action_log_prob, value, rollouts.rewards[step],
                           rollouts.masks[step], rollouts.bad_masks[step])

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            raise Exception('Need to verify that Gail data is compatible with encodings')
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and save_dir != "":
            save_path = os.path.join(save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if args.log_dir and j % args.log_interval == 0 and len(epinfo) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            fps = int(total_num_steps / (end - start))
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, mean win-ratio {:.1f}\n"
                .format(j, total_num_steps,
                        fps,
                        len(epinfo), np.mean([i['r'] for i in epinfo]),
                        np.median([i['r'] for i in epinfo]), np.mean([i['t'] for i in epinfo])))
            writer.add_scalar("Episodes/eprewmean", safemean([i['r'] for i in epinfo]), total_num_steps)
            writer.add_scalar("Episodes/eplenmean", safemean([i['l'] for i in epinfo]), total_num_steps)
            writer.add_scalar("Episodes/eplvlsmean", safemean([i['t'] for i in epinfo]), total_num_steps)
            writer.add_scalar("RL/policy_entropy", dist_entropy, total_num_steps)
            writer.add_scalar("RL/value_loss", value_loss, total_num_steps)
            writer.add_scalar("RL/action_loss", action_loss, total_num_steps)
            writer.add_scalar("Performance/fps", fps, total_num_steps)
            writer.add_scalar("Performance/total_timesteps", total_num_steps, total_num_steps)

        if (args.eval_interval is not None and len(epinfo) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
