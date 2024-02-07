import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as tf
from collections import namedtuple

RSSMState = namedtuple('RSSMState', ['logit', 'stoch', 'h']) #stoch can be z or z^

class RSSM(nn.Module):
    def __init__(self, obs_emb_size, rssm_state_size, action_size, game_type = 'minatari', device='cuda'):
        # action_size, la size delle azioni del env
        # rssm_state_size, la size che vogliamo impostare dell'rnn (size di h)
        # obs_emb_size, la size della z

        nn.Module.__init__(self)
        self.device = device
        self.game_type = game_type

        self.action_size = action_size
        self.obs_emb_size = obs_emb_size
        self.rssm_state_size = rssm_state_size #h_t size
        self.activation = nn.ELU

        if game_type == 'minatari':
            self.category_size = 20
            self.class_size = 20
            self.rssm_hidden_size = self.rssm_state_size
        else:
            self.category_size = 32
            self.class_size = 32
            self.rssm_hidden_size = self.rssm_state_size//2

        
        
        self.z_size = self.category_size*self.class_size # size of z and z^
        #print("obsembedsize", self.obs_emb_size)

        '''print(type(self.z_size), type(action_size), type(rssm_state_size))
        print(self.z_size, action_size, rssm_state_size)
        print('-'*50)
        print('self.rssm_state_size  self.obs_emb_size  self.rssm_state_size')
        print(type(self.rssm_state_size), type(self.obs_emb_size), type(self.rssm_state_size))
        print(self.rssm_state_size, self.obs_emb_size, self.rssm_state_size)
        print('-'*50)
'''
        self.fc_embed_state_action = nn.Sequential(nn.Linear(self.z_size + self.action_size, self.rssm_state_size), self.activation()) #from dreamerv1
        
        self.rnn = nn.GRUCell(self.rssm_state_size, self.rssm_state_size)

        self.fc_prior = nn.Sequential(nn.Linear(self.rssm_state_size, self.rssm_hidden_size),
                                     self.activation(),
                                     nn.Linear(self.rssm_hidden_size, self.z_size)) #dreamerv1

        self.fc_posterior = nn.Sequential(nn.Linear(self.rssm_state_size+self.obs_emb_size, self.rssm_hidden_size),
                                     self.activation(),
                                     nn.Linear(self.rssm_hidden_size, self.z_size)) #dreamerv1

    def init_state(self, batch_size):
        rssm_state = RSSMState(
                        torch.zeros(batch_size, self.z_size).to(self.device), #logits
                        torch.zeros(batch_size, self.z_size).to(self.device), #stochs
                        torch.zeros(batch_size, self.rssm_state_size).to(self.device), #h
                    )
        return rssm_state


    def rssm_observe(self, prev_action, prev_rssm_state, prev_nonterm, obs_embed): #obs_embed relative to x_t in paper
        prior_rssm_state = self.rssm_imagine(prev_action, prev_rssm_state, prev_nonterm)
        h = prior_rssm_state.h
        embed = torch.cat([h, obs_embed], dim=-1)

        posterior_logit = self.fc_posterior(embed)
        stats = {'logit':posterior_logit}
        posterior_stoch_state = self.get_stoch_state(stats)
        posterior_rssm_state = RSSMState(posterior_logit, posterior_stoch_state, h)
        return prior_rssm_state, posterior_rssm_state
    
    def rssm_imagine(self, prev_action, prev_rssm_state, nonterms=True):
        embed = torch.cat([prev_rssm_state.stoch*nonterms, prev_action],dim=-1)
        state_action_embed = self.fc_embed_state_action(embed)
        h = self.rnn(state_action_embed, prev_rssm_state.h*nonterms) #h or h^ based on z or z^

        prior_logit = self.fc_prior(h)
        stats = {'logit':prior_logit}
        prior_stoch_state = self.get_stoch_state(stats) #categorical z^
        prior_rssm_state = RSSMState(prior_logit, prior_stoch_state, h)
        return prior_rssm_state

    def rollout_imagination(self, horizon:int, actor:nn.Module, prev_rssm_state):
        rssm_state = prev_rssm_state
        next_rssm_states = []
        action_entropy = []
        imag_log_probs = []
        for t in range(horizon):
            action, action_dist = actor((self.get_model_state(rssm_state)).detach())
            rssm_state = self.rssm_imagine(action, rssm_state)
            next_rssm_states.append(rssm_state)
            action_entropy.append(action_dist.entropy())
            imag_log_probs.append(action_dist.log_prob(torch.round(action.detach())))

        next_rssm_states = self.rssm_stack_states(next_rssm_states, dim=0)
        action_entropy = torch.stack(action_entropy, dim=0)
        imag_log_probs = torch.stack(imag_log_probs, dim=0)
        return next_rssm_states, imag_log_probs, action_entropy

    def rollout_observation(self, seq_len:int, obs_embed: torch.Tensor, action: torch.Tensor, nonterms: torch.Tensor, prev_rssm_state):
        priors = []
        posteriors = []
        for t in range(seq_len):
            prev_action = action[t]*nonterms[t] #nonterm = 1 if state is non terminal
            prior_rssm_state, posterior_rssm_state = self.rssm_observe(prev_action, prev_rssm_state, nonterms[t], obs_embed[t])
            priors.append(prior_rssm_state)
            posteriors.append(posterior_rssm_state)
            prev_rssm_state = posterior_rssm_state
        prior = self.rssm_stack_states(priors, dim=0)
        post = self.rssm_stack_states(posteriors, dim=0)
        return prior, post # z^ and z?
    
    def get_dist(self, rssm_state):
        shape = rssm_state.logit.shape
        logit = torch.reshape(rssm_state.logit, shape = (*shape[:-1], self.category_size, self.class_size))
        #Gaussians instead of categoricals
        # std = tf.softplus(logit) + 0.1
        #  td.independent.Independent(td.Normal(logit, std), 1)
        return td.Independent(td.OneHotCategoricalStraightThrough(logits=logit), 1) #categorical variables?

    def get_stoch_state(self, stats):
        logit = stats['logit']
        shape = logit.shape
        logit = torch.reshape(logit, shape = (*shape[:-1], self.category_size, self.class_size))
        dist = torch.distributions.OneHotCategorical(logits=logit)        
        stoch = dist.sample() #algorithm 1 in paper
        # print(stoch.shape)
        stoch += dist.probs - dist.probs.detach() #algorithm 1 in paper
        stoch = torch.flatten(stoch, start_dim=-2, end_dim=-1)
        # print(stoch.shape)
        # print(stoch)
        return stoch

    #Gaussians instead of categoricals
    # def get_stoch_state(self, stats):
    #     logit = stats['logit']
    #     shape = logit.shape
    #     logit = torch.reshape(logit, shape = (*shape[:-1], self.category_size, self.class_size))
      
    #     std = tf.softplus(logit) + 0.1
    #     dist = td.Normal(logit, std)
    #     stoch = dist.rsample()
    #     stoch = torch.flatten(stoch, start_dim=-2, end_dim=-1)
    #     # stoch = dist.sample() #algorithm 1 in paper
    #     # stoch += dist.probs - dist.probs.detach() #algorithm 1 in paper
    #     return stoch
    
    def rssm_stack_states(self, rssm_states, dim):
        return RSSMState(
            torch.stack([state.logit for state in rssm_states], dim=dim),
            torch.stack([state.stoch for state in rssm_states], dim=dim),
            torch.stack([state.h for state in rssm_states], dim=dim),
        )

    def get_model_state(self, rssm_state):
        return torch.cat((rssm_state.h, rssm_state.stoch), dim=-1) #h e z concatenate (oppure h^ e z^ dipende dalla situazione)
    
    def rssm_detach(self, rssm_state):
        return RSSMState(
            rssm_state.logit.detach(),  
            rssm_state.stoch.detach(),
            rssm_state.h.detach(),
        )

    def rssm_seq_to_batch(self, rssm_state, batch_size, seq_len):
        return RSSMState(
            seq_to_batch(rssm_state.logit[:seq_len], batch_size, seq_len),
            seq_to_batch(rssm_state.stoch[:seq_len], batch_size, seq_len),
            seq_to_batch(rssm_state.h[:seq_len], batch_size, seq_len)
        )
        
        
    def rssm_batch_to_seq(self, rssm_state, batch_size, seq_len):
        return RSSMState(
            batch_to_seq(rssm_state.logit, batch_size, seq_len),
            batch_to_seq(rssm_state.stoch, batch_size, seq_len),
            batch_to_seq(rssm_state.h, batch_size, seq_len)
        )

    @property
    def modelstate_size(self):
        #ritorna la size che ha l'input del decoder
        return self.rssm_state_size + self.z_size #self.obs_emb_size + 
    

def seq_to_batch(sequence_data, batch_size, seq_len):
    """
    converts a sequence of length L and batch_size B to a single batch of size L*B
    """
    shp = tuple(sequence_data.shape)
    batch_data = torch.reshape(sequence_data, [shp[0]*shp[1], *shp[2:]])
    return batch_data

def batch_to_seq(batch_data, batch_size, seq_len):
    """
    converts a single batch of size L*B to a sequence of length L and batch_size B
    """
    shp = tuple(batch_data.shape)
    seq_data = torch.reshape(batch_data, [seq_len, batch_size, *shp[1:]])
    return seq_data