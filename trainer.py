import copy
from collections import namedtuple
from inspect import getargspec
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from utils import *
from action_utils import *

Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'value_global', 'episode_mask', 'episode_mini_mask', 'next_state',
                                       'reward', 'misc', 'node_maploss', 'edge_maploss'))#'l0','l1','l2''maploss' 'grid_maploss',

'''
Dyanamic Graph version trainer
'''
class Trainer(object):
    def __init__(self, args, policy_net,  env):
        self.args = args
        self.policy_net = policy_net
        self.env = env
        self.display = False
        self.last_step = False
        self.optimizer = optim.RMSprop(policy_net.parameters(),
            lr = args.lrate, alpha=0.97, eps=1e-6)
        self.params = [p for p in self.policy_net.parameters()]
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=40000, gamma=1)
        #self.device = torch.device('cpu') #'cuda:0' if torch.cuda.is_available else

    def ground_truth_gen(self, env):
        nodes_org = np.concatenate((env.predator_loc, env.prey_loc,env.grid_loc), axis=0)
        adj = np.ones((len(nodes_org), len(nodes_org)))*-1
        id = np.zeros((len(nodes_org), len(nodes_org)))
        obs = np.zeros([len(nodes_org), self.args.vision * 2 + 1, self.args.vision * 2 + 1])
        for i, j in enumerate(nodes_org[:self.args.nagents+1]):
            obs[i] = self.setmargin(obs[i], j)
            obs[i][self.args.vision][self.args.vision] = 1
            for x, y in enumerate(nodes_org):
                adj[i][x] = abs(j[0] - y[0]) + abs(j[1] - y[1])
                adj[x][i] = abs(j[0] - y[0]) + abs(j[1] - y[1])
                adj[x][x] = 0
                id[x][x] = 1
                if abs(j[0] - y[0]) <= 1 and abs(j[1] - y[1]) <= 1:
                    obs[i][self.args.vision + y[0] - j[0]][self.args.vision + y[1] - j[1]] = 1
        obstacle_id = np.concatenate(
            (np.zeros([self.args.nagents + 1, 1]), np.ones([len(nodes_org) - self.args.nagents - 1, 1])), axis=0)
        obs_flatten = obs.reshape([len(nodes_org), (self.args.vision * 2 + 1) * (self.args.vision * 2 + 1)])
        id = np.concatenate((id[:, :self.args.nagents + 1], obstacle_id), axis=1)
        # node_ground_truth = np.concatenate((nodes_org, id, obs_flatten), axis=1)
        node_ground_truth = np.concatenate((nodes_org, id, obs_flatten), axis=1) #
        adj[self.args.nagents,:][adj[self.args.nagents,:]>1] = -1
        adj[:, self.args.nagents][adj[:, self.args.nagents] > 1] = -1
        if adj[:, self.args.nagents].sum()+adj[self.args.nagents, :].sum() <= -25*2:
            adj[self.args.nagents,self.args.nagents] = -1
            node_ground_truth[self.args.nagents,:] = -1
        node_ground_truth = np.array(node_ground_truth)
        for i in range(len(env.observed_obstacle)):
            if env.observed_obstacle[i]==0:
                adj[self.args.nagents+1+i, :] = -1
                adj[:, self.args.nagents + 1 + i] = -1
        return node_ground_truth[:self.args.nagents+1], adj[:self.args.nagents+1, :self.args.nagents+1] #

    def state2graph(self, env):
        node_matrix=[]
        nodes_org = np.concatenate((env.predator_loc,env.prey_loc,env.grid_loc),axis=0)
        adj = [np.zeros( (len(nodes_org), len(nodes_org)))]*len(nodes_org)
        for i,j in enumerate(nodes_org[:self.args.nagents+1]):
            clm = np.zeros(len(nodes_org))
            id = np.zeros((len(nodes_org),len(nodes_org)))
            obs = np.zeros([len(nodes_org), self.args.vision * 2 + 1, self.args.vision * 2 + 1])
            obs[i] = self.setmargin(obs[i], j)
            for x,y in enumerate(nodes_org):
                if i==x:
                    clm[x] = 1
                    adj[i][i][x] = 1
                    obs[x][self.args.vision][self.args.vision]=1
                else:
                    if abs(j[0]-y[0])<=1 and abs(j[1]-y[1])<=1:
                        obs[i][self.args.vision+y[0]-j[0]][self.args.vision+y[1]-j[1]]=1
                        obs[x][self.args.vision + j[0]-y[0]][self.args.vision + j[1]-y[1]] = 1
                        obs[x][self.args.vision][self.args.vision] = 1
                        clm[x] = 1
                        adj[i][i][x] = 1
                        adj[i][x][i] = 1
                    else:
                        clm[x] = 0
                id[x][x] = 1
            obs_flatten = obs.reshape([len(nodes_org), (self.args.vision * 2 + 1)*(self.args.vision * 2 + 1)])
            obstacle_id = np.concatenate((np.zeros([self.args.nagents+1,1]), np.ones([len(nodes_org)-self.args.nagents-1,1])), axis=0)
            obs_flatten[self.args.nagents+1:, :] = -1
            id = np.concatenate((id[:, :self.args.nagents+1], obstacle_id), axis=1)
            node_before_mask = np.concatenate((nodes_org, id, obs_flatten), axis=1)
            # node_before_mask = np.concatenate((nodes_org, id, np.ones([len(nodes_org), 1])), axis=1)
            # mask = np.expand_dims(clm, axis=1).repeat(node_before_mask.shape[1], axis=1)
            node = np.delete(node_before_mask, np.where(clm==0)[0],axis=0)
            adj[i] = np.delete(adj[i], np.where(clm==0)[0],axis=0)
            adj[i] = np.delete(adj[i], np.where(clm == 0)[0], axis=1)
            adj[i] = torch.tensor(adj[i])
            main_idx = -1
            if i!= len(nodes_org)-(1+len(env.grid_loc)):
                for idx in range(node.shape[0]):
                    if (node[:,2:2+self.args.nagents+1+1]==id[i]).all():
                        main_idx = idx
                node_matrix.append((torch.tensor(node), main_idx))
        return node_matrix, adj[:self.args.nagents]

    def setmargin(self, obs, pos):
        if self.args.vision-pos[0]>0:
            for i in range(self.args.vision-pos[0]):
                obs[i,:] = -1
        if self.args.vision-pos[1]>0:
            for j in range(self.args.vision-pos[1]):
                obs[:, j] = -1
        if self.args.vision+pos[0]>self.args.dim-1:
            for i in range(self.args.vision+pos[0]-self.args.dim+1):
                obs[-1-i,:] = -1
        if self.args.vision+pos[1]>self.args.dim-1:
            for j in range(self.args.vision+pos[1]-self.args.dim+1):
                obs[:, -1-j] = -1
        return obs

    def get_episode(self, epoch, episodes):
        ############### data collection for boss son ##################
        ### memo record dd = agent_map.detach().numpy() predator loc and prey loc
        ##################################################################
        tracj = np.ones([self.args.nagents,self.args.dim,self.args.dim])*-1
        comm_history = np.ones([self.args.nagents, self.args.dim, self.args.dim]) * -1
        episode = []
        reset_args = getargspec(self.env.reset).args
        if 'epoch' in reset_args:
            state = self.env.reset(epoch)
        else:
            state = self.env.reset()
        should_display = self.display and self.last_step

        if should_display:
            self.env.display()
        stat = dict()
        info = dict()
        switch_t = -1

        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)
        obs_layer = np.zeros([ self.env.env.true.shape[1], self.env.env.true.shape[2]])
        for t in range(self.args.max_steps):
            misc = dict()
            if t == 0 and self.args.hard_attn and self.args.commnet:
                info['comm_action'] = np.zeros(self.args.nagents, dtype=int)

            # recurrence over time
            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and t == 0:
                    prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])
                # change x to graph as
                node, adj = self.state2graph(self.env.env)
                x = [node, adj, prev_hid]
                action_out, value, value_global, prev_hid, node_decoded, edge_decoded = self.policy_net(x, info) #, grid_decoded

                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:
                        prev_hid = prev_hid.detach()
            else:
                x = state
                action_out, value ,value_global = self.policy_net(x, info)

            action = select_action(self.args, action_out)
            action, actual = translate_action(self.args, self.env, action)

            node_decoded = node_decoded.view(self.args.nagents, self.args.nagents+1, (2+self.args.nagents+1+1+9))
            edge_decoded = edge_decoded.view(self.args.nagents, self.args.nagents+1, self.args.nagents+1) #+self.env.env.ngrid
            # for i in range(len(self.env.env.observed_obstacle)):
            #     if self.env.env.observed_obstacle[i] == 0:
            #         edge_decoded[:, self.args.nagents + 1 + i, :] = -1
            #         edge_decoded[:, :, self.args.nagents + 1 + i] = -1
            # grid_decoded = grid_decoded.view(self.args.nagents, self.args.dim, self.args.dim)
            node_gt, edge_gt = self.ground_truth_gen(self.env.env)
            #obs_layer = obs_layer + self.env.env.true[2, :, :]
            #obs_layer[obs_layer > 0] = 1
            #self.env.env.true[2, :, :] = obs_layer
            # grid_truth = torch.tensor(self.env.env.true[2, :, :][np.newaxis].repeat(self.args.nagents, axis=0))
            # gt = gt[np.newaxis]
            # node_ground_truthg = torch.tensor(node_gt[np.newaxis].repeat(3, axis=0))
            ######## visualize #######
            belief_node = node_decoded.detach().numpy()
            belief_edge = edge_decoded.detach().numpy()
            node_ground_truthg = torch.tensor(node_gt[np.newaxis].repeat(self.args.nagents, axis=0))
            edge_ground_truthg = torch.tensor(edge_gt[np.newaxis].repeat(self.args.nagents, axis=0))
            Loss_func = nn.MSELoss(reduction='sum')
            node_maploss = Loss_func(node_decoded, node_ground_truthg.detach())
            edge_maploss = Loss_func(edge_decoded, edge_ground_truthg.detach())
            # grid_maploss = Loss_func(grid_decoded, grid_truth.detach())
            # maploss = node_maploss + edge_maploss + grid_maploss
            '''
            ############### son dd
            np.save('./data/coordinates/agents' +str(episodes)+'_' + str(t) + '.npy',
                    np.concatenate((self.env.env.predator_loc, self.env.env.prey_loc), axis=0))
            np.save('./data/coordinates/obstacles' + str(episodes) + '_' + str(t) + '.npy',
                    self.env.env.grid_loc)
            np.save('./data/decoder/decoded_node_'+str(episodes)+'_' + str(t) + '.npy', belief_node)
            np.save('./data/decoder/decoded_edge_' + str(episodes) + '_' + str(t) + '.npy', belief_edge)
            # np.save('./data/agent_loc/predator_loc_map'+str(episodes)+'_' + str(t) + '.npy', self.env.env.predator_loc)
            np.save('./data/ground_truth/ground_truth_node'+str(episodes)+'_' + str(t) + '.npy', node_gt)
            np.save('./data/ground_truth/ground_truth_edge' + str(episodes) + '_' + str(t) + '.npy', edge_gt)
            np.save('./data/comm/communication'+str(episodes)+'_' + str(t) + '.npy', action[-1])
            np.save('./data/global_value/value_g'+str(episodes)+'_' + str(t) + '.npy', value_global.detach().numpy())
            
            '''
            value_global = value_global.sum(1).view(value_global.shape[0], 1)
            #record trajectory and comm history
            #for i, p in enumerate(self.env.env.predator_loc):
            #    tracj[i, p[0], p[1]] = t
            #    if comm_history[i, p[0], p[1]] == -1:
            #        comm_history[i, p[0], p[1]] = 0
            #    comm_history[i, p[0], p[1]] = comm_history[i, p[0], p[1]] + action[-1][i]

            next_state, reward, done, info = self.env.step(actual)
            #np.save('./data/gt_value/value_g' + str(episodes) + '_' + str(t) + '.npy', reward)
            next_state = next_state.squeeze().view(1, self.args.nagents, 4, self.args.dim, self.args.dim)
            # store comm_action in info for next step
            if self.args.hard_attn and self.args.commnet:
                info['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(self.args.nagents, dtype=int)

                stat['comm_action'] = stat.get('comm_action', 0) + info['comm_action'][:self.args.nfriendly]
                if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                    stat['enemy_comm']  = stat.get('enemy_comm', 0)  + info['comm_action'][self.args.nfriendly:]


            if 'alive_mask' in info:
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones_like(reward)

            # env should handle this make sure that reward for dead agents is not counted
            # reward = reward * misc['alive_mask']

            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

            done = done or t == self.args.max_steps - 1

            episode_mask = np.ones(reward.shape)
            episode_mini_mask = np.ones(reward.shape)

            if done:
                episode_mask = np.zeros(reward.shape)
            else:
                if 'is_completed' in info:
                    episode_mini_mask = 1 - info['is_completed'].reshape(-1)

            if should_display:
                self.env.display()


            trans = Transition(state, action, action_out, value, value_global, episode_mask, episode_mini_mask, next_state, reward, misc, node_maploss, edge_maploss)
            # trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc, maploss) grid_maploss,

            episode.append(trans)
            state = next_state
            if done:
                break
        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']

        ############# sons data
        # np.save('./data/trajectory_map'+str(episodes)+'.npy', tracj)
        # np.save('./data/communication_history_map'+str(episodes)+'.npy', comm_history)

        if hasattr(self.env, 'reward_terminal'):
            reward = self.env.reward_terminal()
            # We are not multiplying in case of reward terminal with alive agent
            # If terminal reward is masked environment should do
            # reward = reward * misc['alive_mask']

            episode[-1] = episode[-1]._replace(reward = episode[-1].reward + reward)
            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]


        # stat['min_steps'] = self.env.env.min_steps # pretrain vision
        # stat['min_steps'] = 0 # pretrain vision 2
        if hasattr(self.env, 'get_stat'):
            merge_stat(self.env.get_stat(), stat)
        return (episode, stat)

    def compute_grad(self, batch, ep):
        stat = dict()
        num_actions = self.args.num_actions
        dim_actions = self.args.dim_actions

        n = self.args.nagents
        ng = self.args.obstacles
        batch_size = len(batch.state)

        rewards = torch.Tensor(batch.reward)
        episode_masks = torch.Tensor(batch.episode_mask)
        episode_mini_masks = torch.Tensor(batch.episode_mini_mask)
        actions = torch.Tensor(batch.action)
        actions = actions.transpose(1, 2).view(-1, n, dim_actions)
        '''
        rewards = rewards.to(self.device)
        episode_masks = episode_masks.to(self.device)
        episode_mini_masks = episode_mini_masks.to(self.device)
        actions = actions.to(self.device)
        '''
        # old_actions = torch.Tensor(np.concatenate(batch.action, 0))
        # old_actions = old_actions.view(-1, n, dim_actions)
        # print(old_actions == actions)

        # can't do batch forward.
        values = torch.cat(batch.value, dim=0)
        values_g = torch.cat(batch.value_global, dim=0)
        action_out = list(zip(*batch.action_out))
        action_out = [torch.cat(a, dim=0) for a in action_out]

        alive_masks = torch.Tensor(np.concatenate([item['alive_mask'] for item in batch.misc])).view(-1)
        #alive_masks = alive_masks.to(self.device)
        coop_returns = torch.Tensor(batch_size, n)#.cuda()
        ncoop_returns = torch.Tensor(batch_size, n)#.cuda()
        returns = torch.Tensor(batch_size, n)#.cuda()
        deltas = torch.Tensor(batch_size, n)#.cuda()
        advantages = torch.Tensor(batch_size, n)#.cuda()
        values = values.view(batch_size, n)
        values_g = values_g.view(batch_size, n)

        prev_coop_return = 0
        prev_ncoop_return = 0
        prev_value = 0
        prev_advantage = 0

        for i in reversed(range(rewards.size(0))):
            coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * episode_masks[i]
            ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]

            prev_coop_return = coop_returns[i].clone()
            prev_ncoop_return = ncoop_returns[i].clone()

            returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
                        + ((1 - self.args.mean_ratio) * ncoop_returns[i])


        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i]

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()

        if self.args.continuous:
            action_means, action_log_stds, action_stds = action_out
            log_prob = normal_log_density(actions, action_means, action_log_stds, action_stds)
        else:
            log_p_a = [action_out[i].view(-1, num_actions[i]) for i in range(dim_actions)]
            actions = actions.contiguous().view(-1, dim_actions)

            if self.args.advantages_per_action:
                log_prob = multinomials_log_densities(actions, log_p_a)
            else:
                log_prob = multinomials_log_density(actions, log_p_a)

        map_loss_0 = torch.stack(batch.node_maploss, dim=0)
        map_loss_1 = torch.stack(batch.edge_maploss, dim=0)
        # map_loss_2 = torch.stack(batch.grid_maploss, dim=0)
        # imitation_loss = torch.stack(batch.action_dist, dim=0)

        if self.args.advantages_per_action:
            action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
            action_loss *= alive_masks.unsqueeze(-1)
        else:
            action_loss = -advantages.view(-1) * log_prob.squeeze()
            action_loss *= alive_masks

        map_loss_m0 = map_loss_0.sum()
        map_loss_m1 = map_loss_1.sum()
        #map_loss_m2 = map_loss_2.sum()
        # imitation_loss_sum = imitation_loss.sum()
        # stat['distance_loss'] = imitation_loss_sum.item()

        action_loss = action_loss.sum()
        stat['action_loss'] = action_loss.item()

        # value loss term
        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        value_loss *= alive_masks
        value_loss = value_loss.sum()
        # gloable value loss term
        targets_g = returns.sum(1).view(batch_size,1)
        # value_loss_g = (values_g/self.args.nagents - targets_g/self.args.nagents).pow(2).view(-1)
        value_loss_g = (values_g - targets_g).pow(2).view(-1) # Feb setting
        value_loss_g *= alive_masks
        value_loss_g = value_loss_g.sum()

        stat['value_loss'] = value_loss.item()
        stat['value_loss_g'] = (value_loss_g/self.args.nagents).item()

        # map_loss = (map_loss_m0/ ((n+1)*(2+n+2+9)) + map_loss_m1/(n+1)**2 )   
        map_loss = (map_loss_m0/ ((n+1+ng)*(2+n+1+9)) + map_loss_m1/(n+1+ng)**2 ) # Feb setting  
        stat['map_loss'] = map_loss.item()
        loss = action_loss + self.args.value_coeff * (value_loss) + self.args.value_coeff * (value_loss_g/self.args.nagents) + 0.1*map_loss 


        if not self.args.continuous:
            # entropy regularization term
            entropy = 0
            for i in range(len(log_p_a)):
                entropy -= (log_p_a[i] * log_p_a[i].exp()).sum()
            stat['entropy'] = entropy.item()
            if self.args.entr > 0:
                loss -= self.args.entr * entropy


        stat['loss'] = loss.item()
        loss.backward()

        return stat

    def run_batch(self, epoch):
        batch = []
        self.stats = dict()
        self.stats['num_episodes'] = 0
        while len(batch) < self.args.batch_size: # commended for data collection
        # while self.stats['num_episodes'] < 50:
            if self.args.batch_size - len(batch) <= self.args.max_steps:
                self.last_step = True
            episode, episode_stat = self.get_episode(epoch, self.stats['num_episodes'])
            merge_stat(episode_stat, self.stats)
            self.stats['num_episodes'] += 1
            batch += episode

        self.last_step = False
        self.stats['num_steps'] = len(batch)
        batch = Transition(*zip(*batch))
        return batch, self.stats

    # only used when nprocesses=1
    def train_batch(self, epoch):
        batch, stat = self.run_batch(epoch)
        self.optimizer.zero_grad()

        s = self.compute_grad(batch, epoch)
        merge_stat(s, stat)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= stat['num_steps']
        self.optimizer.step()

        return stat

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)


#####  heuristic for Predator Prey
    def chasing_prey(self, node):
        for i in range(node.shape[0]):
            if node[i].sum()<=0:
                a=1
