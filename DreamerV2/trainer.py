from world_model import *
from actor_critic import *
import torch.optim as optim
import os
from minatar.gui import GUI

class Trainer():
    def __init__(self, env_name, action_size, game_type = 'minatari', device='cuda'):
        self.device = device
        self.game_type = game_type
        self.env_name = env_name
        self.batch_size = 50
        self.seq_len = 50

        if self.game_type == 'minatari':
            self.obs_shape = (6,10,10) #Dimensione non modulare, va cambiata in base al gioco (asterix = (4,10,10))
        else:
            self.obs_shape = (1,64,64)
        self.action_size = action_size
        self.collect_intervals = 5
        self.grad_clip_norm = 100.0
        self.recent_scores = []
        self.recent_steps = []
        #parametri che vengono usati nella funzione train
        self.train_steps = int(5e7)
        self.train_every = int(50)
        self.slow_target_update = int(100)
        self.save_every = int(5e4) #int(5e4)

        self.seed_steps = int(4000) #4000
        self.slow_target_fraction = 1.00
        self.model_dir = 'saved_models/' + self.env_name + '/'
        makedir(self.model_dir)
        print("trainer",self.obs_shape)
        self.wrld_model = WorldModel(self.batch_size, self.seq_len, self.obs_shape, self.action_size, device=self.device, game_type=self.game_type)
        self.actor_critic = ActorCritic(self.wrld_model, 100, 100, self.action_size, batch_size=self.batch_size, seq_len=self.seq_len, device=self.device)

        self.wrld_model_lr = 2e-4
        self.actor_lr = 4e-5
        self.value_lr = 1e-4

        self.model_optimizer = optim.Adam(get_parameters(self.wrld_model.get_modules), self.wrld_model_lr, eps=1e-5, weight_decay=1e-6)
        self.actor_optimizer = optim.Adam(get_parameters([self.actor_critic.action_model]), self.actor_lr, eps=1e-5, weight_decay=1e-6)
        self.value_optimizer = optim.Adam(get_parameters([self.actor_critic.value_model]), self.value_lr, eps=1e-5, weight_decay=1e-6)

    def train_batch(self):
        for i in range(self.collect_intervals):
            obs, actions, rewards, terms = self.wrld_model.buffer.sample()
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)                         #t, t+seq_len 
            actions = torch.tensor(actions, dtype=torch.float32).to(self.device)                 #t-1, t+seq_len-1
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(-1)   #t-1 to t+seq_len-1
            nonterms = torch.tensor(1-terms, dtype=torch.float32).to(self.device).unsqueeze(-1)  #t-1 to t+seq_len-1

            #print('obs shape', obs.shape)
            model_loss, kl_loss, obs_loss, reward_loss, pcont_loss, prior_dist, post_dist, posterior = self.wrld_model.representation_loss(obs, actions, rewards, nonterms)
            self.model_optimizer.zero_grad()
            model_loss.backward()
            torch.nn.utils.clip_grad_norm_(get_parameters(self.wrld_model.get_modules), self.grad_clip_norm)
            self.model_optimizer.step()

            actor_loss, value_loss = self.actor_critic.actorcritic_loss(posterior)
            self.actor_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            actor_loss.backward()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(get_parameters([self.actor_critic.action_model]), self.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(get_parameters([self.actor_critic.value_model]), self.grad_clip_norm)
            self.actor_optimizer.step()
            self.value_optimizer.step()

            #with torch.no_grad():
            #    prior_ent = torch.mean(prior_dist.entropy())
            #    post_ent = torch.mean(post_dist.entropy())
        
    def train(self, env, resume_training):
        #when resuming the training we load the saved arrays
        if resume_training > 0:
            print('loading buffers...')
            self.wrld_model.buffer.load_buffer(self.model_dir)
            print("loading model's dicts...")
            self.load_save_dict(self.model_dir + 'models_' + str(resume_training) + '.pth')
        else:
            self.collect_experience(env)

        obs, score = env.reset(), 0
        #print("obs_shape",obs.shape)
        done = False
        prev_rssmstate = self.wrld_model.rssm.init_state(1)
        prev_action = torch.zeros(1, self.action_size).to(self.device)
        # episode_actor_ent = []
        scores = []
        best_mean_score = 0
        best_save_path = os.path.join(self.model_dir, 'models_best.pth')

        for iter in range(resume_training + 1, self.train_steps):
            if self.game_type == 'minatari':
                obs = torch.tensor(obs, dtype=torch.float32)
            if iter%self.train_every == 0:
                self.train_batch() #also update train_metrics
            if iter%self.slow_target_update == 0:
                self.update_target()                
            if iter%self.save_every == 0:
                
                self.f_scores = open(f'scores/{self.env_name}_scores.txt', 'a+')
                self.f_scores.write(','.join(self.recent_scores)+',')
                self.f_scores.close()
                self.recent_scores = []

                self.f_steps = open(f'scores/{self.env_name}_steps.txt', 'a+')
                self.f_steps.write(','.join(self.recent_steps)+',')
                self.f_steps.close()
                self.recent_steps = []

                self.save_model(iter)
                self.wrld_model.buffer.save_buffer(self.model_dir)
                
            with torch.no_grad():
                #embed = self.wrld_model.encoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device))  #x_t
                embed = self.wrld_model.encoder(obs.unsqueeze(0).to(self.device))  #x_t
                #print("trainer embed", embed.shape)
                _, posterior_rssm_state = self.wrld_model.rssm.rssm_observe(prev_action, prev_rssmstate, not done, embed) #h_t e z_t (in una tupla)    
                model_state = self.wrld_model.rssm.get_model_state(posterior_rssm_state) #h_t+z_t
                #print('la model state size reale e: ', model_state.shape)
                action, action_dist = self.actor_critic.action_model(model_state) #azione con gradiente e distribuzione di probabilita' delle azioni
                action = self.actor_critic.action_model.explore(action, iter).detach()
                #action_ent = torch.mean(action_dist.entropy()).item()
                #episode_actor_ent.append(action_ent)

            next_obs, rew, done, _ = env.step(action.squeeze(0).cpu().numpy()) 
            # print(done, 'DONEEEEEEEEEEE')
            score += rew
            if done:
                self.recent_scores.append(str(score))
                self.recent_steps.append(str(iter))
                # self.f_scores.write(f'{score},')
                # self.f_scores.flush()
                print("game ended with score:", score)
                self.wrld_model.buffer.add(obs, action.squeeze(0).cpu().numpy(), rew, done)
                scores.append(score)
                if len(scores)>50:
                    scores.pop(0)
                    current_average = np.mean(scores)
                    print('current_average', current_average, 'step', iter)
                    if current_average>best_mean_score:
                        best_mean_score = current_average 
                        print('saving best model with mean score : ', best_mean_score, '\tat step : ', iter, '/ ', self.train_steps)
                        save_dict = self.get_save_dict()
                        torch.save(save_dict, best_save_path)
                
                obs, score = env.reset(), 0
                done = False
                prev_rssmstate = self.wrld_model.rssm.init_state(1)
                prev_action = torch.zeros(1, self.action_size).to(self.device)
                # episode_actor_ent = []
            else:
                self.wrld_model.buffer.add(obs, action.squeeze(0).detach().cpu().numpy(), rew, done)
                obs = next_obs
                prev_rssmstate = posterior_rssm_state
                prev_action = action

    def collect_experience(self, env):
        s, done  = env.reset(), False 
        for i in range(self.seed_steps):
            a = env.env.action_space.sample()
            ns, r, done, _ = env.step(a)
            if done:
                self.wrld_model.buffer.add(s,a,r,done)
                s, done  = env.reset(), False 
            else:
                self.wrld_model.buffer.add(s,a,r,done)
                s = ns    
    
    def update_target(self):
        mix = self.slow_target_fraction
        for param, target_param in zip(self.actor_critic.value_model.parameters(), self.actor_critic.target_value_model.parameters()):
            target_param.data.copy_(mix * param.data + (1 - mix) * target_param.data)

    def save_model(self, iter):
        save_dict = self.get_save_dict()
        model_dir = self.model_dir
        save_path = os.path.join(model_dir, 'models_%d.pth' % iter)
        torch.save(save_dict, save_path)

    def get_save_dict(self):
        return {
            "RSSM": self.wrld_model.rssm.state_dict(),
            "ObsEncoder": self.wrld_model.encoder.state_dict(),
            "ObsDecoder": self.wrld_model.decoder.state_dict(),
            "RewardDecoder": self.wrld_model.reward_model.state_dict(),
            "ActionModel": self.actor_critic.action_model.state_dict(),
            "ValueModel": self.actor_critic.value_model.state_dict(),
            "DiscountModel": self.wrld_model.discount_model.state_dict(),
        }
    
    def load_save_dict(self, saved_dict_path):
        saved_dict = torch.load(saved_dict_path, map_location=self.device)
        self.wrld_model.rssm.load_state_dict(saved_dict["RSSM"])
        self.wrld_model.encoder.load_state_dict(saved_dict["ObsEncoder"])
        self.wrld_model.decoder.load_state_dict(saved_dict["ObsDecoder"])
        self.wrld_model.reward_model.load_state_dict(saved_dict["RewardDecoder"])
        self.actor_critic.action_model.load_state_dict(saved_dict["ActionModel"])
        self.actor_critic.value_model.load_state_dict(saved_dict["ValueModel"])
        self.wrld_model.discount_model.load_state_dict(saved_dict['DiscountModel'])

    def evaluation(self, env, saved_dict_path, games):
        eval_scores = []
        # self.f_eval_scores = open(f'scores/{self.env_name}_eval_scores.txt', 'a+')
        # self.f_eval_scores.write(','.join(self.eval_scores)+',')
        # self.f_eval_scores.close()

        self.load_save_dict(saved_dict_path)

        #print("obs_shape",obs.shape)
        done = False
        prev_rssmstate = self.wrld_model.rssm.init_state(1)
        prev_action = torch.zeros(1, self.action_size).to(self.device)
        game = 0
        self.wrld_model.encoder.eval()
        self.wrld_model.rssm.eval()
        self.wrld_model.decoder.eval()
        self.wrld_model.reward_model.eval()
        self.actor_critic.action_model.eval()
        self.actor_critic.value_model.eval()
        self.wrld_model.discount_model.eval()

        g = GUI(env.env.game_name(), env.env.n_channels)
        def func():
        #     g.display_state(env.env.env.state())
            return
        obs, score = env.reset(), 0

        while game < games:
            done = False
            while not done:
                g.update(0, func) 
                if self.game_type == 'minatari':
                    obs = torch.tensor(obs, dtype=torch.float32)

                with torch.no_grad():
                    #embed = self.wrld_model.encoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device))  #x_t
                    embed = self.wrld_model.encoder(obs.unsqueeze(0).to(self.device))  #x_t
                    #print("trainer embed", embed.shape)
                    _, posterior_rssm_state = self.wrld_model.rssm.rssm_observe(prev_action, prev_rssmstate, not done, embed) #h_t e z_t (in una tupla)    
                    model_state = self.wrld_model.rssm.get_model_state(posterior_rssm_state) #h_t+z_t
                    #print('la model state size reale e: ', model_state.shape)
                    action, action_dist = self.actor_critic.action_model(model_state) #azione con gradiente e distribuzione di probabilita' delle azioni
                    action = self.actor_critic.action_model.explore(action, 0, eval = True).detach()
                    #action_ent = torch.mean(action_dist.entropy()).item()
                    #episode_actor_ent.append(action_ent)

                g.display_state(env.env.env.state())
                next_obs, rew, done, _ = env.step(action.squeeze(0).cpu().numpy()) 
                env.render()
                g.update(50, func)
                score += rew
                if done:
                    
                    eval_scores.append(score)
                    # self.f_scores.write(f'{score},')
                    # self.f_scores.flush()
                    print(f"game {game} ended with score:", score)
                    game += 1
                    self.wrld_model.buffer.add(obs, action.squeeze(0).cpu().numpy(), rew, done)
                    eval_scores.append(score)
                    
                    obs, score = env.reset(), 0
                    prev_rssmstate = self.wrld_model.rssm.init_state(1)
                    prev_action = torch.zeros(1, self.action_size).to(self.device)
                    
                else:
                    self.wrld_model.buffer.add(obs, action.squeeze(0).detach().cpu().numpy(), rew, done)
                    obs = next_obs
                    prev_rssmstate = posterior_rssm_state
                    prev_action = action
        
        print("average score: ", np.mean(eval_scores))

def makedir(path):
    print('creating directory', path)
    if not os.path.exists(path):
        os.makedirs(path)
        print('directory created successfully')
    else:
        print('directory', path, 'already exists')
    