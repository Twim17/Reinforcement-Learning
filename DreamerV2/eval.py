from utils import *
from obs import *
from dense import *
from actor_critic import *
from rssm import *
from env import *

class Evaluator(object):
    def __init__(
        self, 
        env,
        device,
    ):
        self.device = device
        self.env = env
        self.eval_render = True
        
        self.obs_shape = (1,64,64)
        self.action_size = self.env.act_shape

        self.action_model_out_size = (self.action_size,) #perche' altrimenti non funziona bene il len(out_size) nel densemodel
        self.action_model_hidden_size = 100
        # self.action_model_in_size = self.world_model.rssm.modelstate_size
        self.action_size = self.action_size
        # self.value_model_in_size = self.rssm.modelstate_size
        

        self.model_dir = 'saved_models/'



    def load_model(self, model_path):
        saved_dict = torch.load(model_path)
        
        self.ObsEncoder = ObsEncoder(self.obs_shape, 1024).to(self.device).eval()
        self.rssm = RSSM(self.ObsEncoder.embed_size, 600, self.action_size, device=self.device).to(self.device).eval()
        self.ObsDecoder = ObsDecoder(self.rssm.modelstate_size).to(self.device).eval()
        
        self.action_model = ActionModel(self.rssm.modelstate_size, self.action_model_out_size, self.action_model_hidden_size, None).to(self.device).eval()
    
        self.rssm.load_state_dict(saved_dict["RSSM"])
        self.ObsEncoder.load_state_dict(saved_dict["ObsEncoder"])
        self.ObsDecoder.load_state_dict(saved_dict["ObsDecoder"])
        self.action_model.load_state_dict(saved_dict["ActionModel"])

    def eval_saved_agent(self, env, model_path):
        
        self.load_model(model_path)
        eval_episode = 4
        eval_scores = []             

        for e in range(eval_episode):
            obs, score = env.reset(), 0   
            
            done = False
            prev_rssmstate = self.rssm.init_state(1)
            prev_action = torch.zeros(1, self.action_size).to(self.device)
            while not done:
                with torch.no_grad():
                    embed = self.ObsEncoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device))    
                    _, posterior_rssm_state = self.rssm.rssm_observe(prev_action, prev_rssmstate, not done, embed)
                    model_state = self.rssm.get_model_state(posterior_rssm_state)
                    action, _ = self.action_model(model_state)
                    prev_rssmstate = posterior_rssm_state
                    prev_action = action
    
                next_obs, rew, done, _ = env.step(action.squeeze(0).cpu().numpy())
                
                # next_obs, rew, done = gfunc(e,action)
                # if self.eval_render:
                #     env.render()
                score += rew
                obs = next_obs
            eval_scores.append(score)

            
        print('average evaluation score for model at ' + model_path + ' = ' +str(np.mean(eval_scores)))
        env.close()

        return np.mean(eval_scores)

def main():
    env_name = 'Breakout-v0' 
    device='cuda' 

    env = GymAtari(env_name, render_mode='human')

    model_dir = "./saved_models"


    ev = Evaluator(env, device)
    ev.eval_saved_agent(env, model_dir + '/models_best.pth')
    
if __name__=='__main__':
    
    main()