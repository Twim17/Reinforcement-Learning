from buffer import *
from rssm import *
from obs import *
from dense import *
from torchsummary import summary

class WorldModel():
    def __init__(self, batch_size, seq_len, obs_shape, action_size, game_type = 'minatari', device='cuda'):
        self.batch_size = batch_size
        self.game_type = game_type
        self.seq_len = seq_len
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.loss_scale_discount = 5.0
        self.loss_scale_kl = 0.1
        self.discount = 0.999
        self.device = device
        self.build_models()
        
    def build_models(self):
        print("wrld model", self.obs_shape)
        self.buffer = Buffer(int(2e6), self.obs_shape, self.action_size, self.batch_size, self.seq_len, obs_type=np.float32, game_type=self.game_type)
        if self.game_type == 'minatari':
            self.encoder = ObsEncoder(self.obs_shape, out_shape=200, game_type=self.game_type).to(self.device)
            self.rssm = RSSM(self.encoder.out_shape, 200, self.action_size, device=self.device, game_type=self.game_type).to(self.device)
            self.decoder = ObsDecoder(self.rssm.modelstate_size, game_type=self.game_type).to(self.device)
            self.reward_model = DenseModel(self.rssm.modelstate_size, (1,), 400, 'normal').to(self.device)
            self.discount_model = DenseModel(self.rssm.modelstate_size, (1,), 400, 'binary').to(self.device)

        else:
            self.encoder = ObsEncoder(self.obs_shape, 1024, game_type=self.game_type).to(self.device)
            self.rssm = RSSM(self.encoder.embed_size, 600, self.action_size, device=self.device).to(self.device)
            self.decoder = ObsDecoder(self.rssm.modelstate_size).to(self.device)
            self.reward_model = DenseModel(self.rssm.modelstate_size, (1,), 400, 'normal').to(self.device)
            self.discount_model = DenseModel(self.rssm.modelstate_size, (1,), 400, 'binary').to(self.device)
        #for model in self.get_modules:
        #    summary(model)
        #summary(self.decoder)

    @property
    def get_modules(self):
        return [self.encoder, self.rssm, self.reward_model, self.decoder, self.discount_model]

    def representation_loss(self, obs, actions, rewards, nonterms):
        embed = self.encoder(obs)                                         #t to t+seq_len   
        prev_rssm_state = self.rssm.init_state(self.batch_size)   
        prior, posterior = self.rssm.rollout_observation(self.seq_len, embed, actions, nonterms, prev_rssm_state)
        post_modelstate = self.rssm.get_model_state(posterior)               #t to t+seq_len   
        obs_dist = self.decoder(post_modelstate[:-1])                     #t to t+seq_len-1  
        reward_dist = self.reward_model(post_modelstate[:-1])               #t to t+seq_len-1  
        pcont_dist = self.discount_model(post_modelstate[:-1])                #t to t+seq_len-1   
        
        obs_loss = self.obs_loss(obs_dist, obs[:-1]) #da levare per esperimento
        reward_loss = self.reward_loss(reward_dist, rewards[1:])
        pcont_loss = self.pcont_loss(pcont_dist, nonterms[1:])
        prior_dist, post_dist, div = self.kl_loss(prior, posterior) #da levare per esperimento

        # MODELLO BASE 
        model_loss = self.loss_scale_kl * div + reward_loss + obs_loss + self.loss_scale_discount*pcont_loss
        # MODELLO SENZA OBS GRADIENT 
        # model_loss = self.loss_scale_kl * div + reward_loss + self.loss_scale_discount*pcont_loss
        # MODELLO SENZA KL model_loss = reward_loss + obs_loss + self.loss_scale_discount*pcont_loss
        return model_loss, div, obs_loss, reward_loss, pcont_loss, prior_dist, post_dist, posterior
    
    def kl_loss(self, prior, posterior, kl_balance_scale=0.8):
        prior_dist = self.rssm.get_dist(prior)
        post_dist = self.rssm.get_dist(posterior)
        
        alpha = kl_balance_scale 
        #Algorithm 2
        kl_lhs = torch.mean(torch.distributions.kl.kl_divergence(self.rssm.get_dist(self.rssm.rssm_detach(posterior)), prior_dist))
        kl_rhs = torch.mean(torch.distributions.kl.kl_divergence(post_dist, self.rssm.get_dist(self.rssm.rssm_detach(prior))))
        
        kl_loss = alpha*kl_lhs + (1-alpha)*kl_rhs #KL BALANCING
        kl_loss = torch.mean(torch.distributions.kl.kl_divergence(post_dist, prior_dist))
        
        return prior_dist, post_dist, kl_loss
    
    def obs_loss(self, obs_dist, obs):
        obs_loss = -torch.mean(obs_dist.log_prob(obs))
        return obs_loss

    def reward_loss(self, reward_dist, rewards):
        reward_loss = -torch.mean(reward_dist.log_prob(rewards))
        return reward_loss
    
    def pcont_loss(self, pcont_dist, nonterms):
        pcont_target = nonterms.float()
        pcont_loss = -torch.mean(pcont_dist.log_prob(pcont_target))
        return pcont_loss