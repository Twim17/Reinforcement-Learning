from action_model import *
from utils import *
from torchsummary import summary

class ActorCritic():
    def __init__(self, world_model, action_model_hidden_size, value_model_hidden_size, action_size, batch_size, seq_len, device='cuda'):
        self.device = device
        self.world_model = world_model
        self.action_model_out_size = (action_size,) #perche' altrimenti non funziona bene il len(out_size) nel densemodel
        self.action_model_hidden_size = action_model_hidden_size
        self.action_model_in_size = self.world_model.rssm.modelstate_size
        self.action_size = action_size
        self.value_model_in_size = self.world_model.rssm.modelstate_size
        self.value_model_out_size = (1,) #perche' altrimenti non funziona bene il len(out_size) nel densemodel
        self.value_model_hidden_size = value_model_hidden_size 

        self.batch_size = batch_size
        self.seq_len = seq_len

        self.epsilon = 0.4
        self.expl_decay = 7000
        self.min_epsilon = 0.05

        self.lambda_ = 0.95
        self.horizon = 10
        self.actor_entropy_scale = 1e-3

        self.world_list = self.world_model.get_modules
        self.build_model()
   
    def build_model(self):
        self.rssm = self.world_model.rssm
        self.action_model = ActionModel(self.action_model_in_size, self.action_model_out_size,
                                        self.action_model_hidden_size, None, self.epsilon, 
                                        self.expl_decay, self.min_epsilon, self.action_size).to(self.device)
        
        self.value_model = DenseModel(self.value_model_in_size, self.value_model_out_size, self.value_model_hidden_size, dist='normal').to(self.device)
        self.target_value_model = DenseModel(self.value_model_in_size, self.value_model_out_size, self.value_model_hidden_size, dist='normal').to(self.device)
        self.target_value_model.load_state_dict(self.value_model.state_dict())
        #summary(self.action_model)
        #summary(self.value_model)
        #summary(self.target_value_model)
    
    def actor_loss(self, imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy):
        #lambda_returns = Q
        lambda_returns = compute_return(imag_reward[:-1], imag_value[:-1], discount_arr[:-1], bootstrap=imag_value[-1], lambda_=self.lambda_)
        
        advantage = (lambda_returns-imag_value[:-1]).detach()
        objective = imag_log_prob[1:].unsqueeze(-1) * advantage

        discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]])
        discount = torch.cumprod(discount_arr[:-1], 0)
        policy_entropy = policy_entropy[1:].unsqueeze(-1)
        actor_loss = -torch.sum(torch.mean(discount * (objective + self.actor_entropy_scale * policy_entropy), dim=1)) #eq 6 in paper
        return actor_loss, discount, lambda_returns

    def value_loss(self, imag_modelstates, discount, lambda_returns): #EQ 4 IN PAPER
        with torch.no_grad():
            value_modelstates = imag_modelstates[:-1].detach()
            value_discount = discount.detach()
            value_target = lambda_returns.detach()

        value_dist = self.value_model(value_modelstates) 
        value_loss = -torch.mean(value_discount*value_dist.log_prob(value_target).unsqueeze(-1))
        return value_loss

    def actorcritic_loss(self, posterior):
        with torch.no_grad():
            batched_posterior = self.rssm.rssm_detach(self.rssm.rssm_seq_to_batch(posterior, self.batch_size, self.seq_len-1))
       
        with FreezeParameters(self.world_list):
            imag_rssm_states, imag_log_prob, policy_entropy = self.rssm.rollout_imagination(self.horizon, self.action_model, batched_posterior)
        
        imag_modelstates = self.rssm.get_model_state(imag_rssm_states)
        with FreezeParameters(self.world_list+[self.value_model]+[self.target_value_model]+[self.world_model.discount_model]):
            imag_reward_dist = self.world_model.reward_model(imag_modelstates)
            imag_reward = imag_reward_dist.mean
            imag_value_dist = self.target_value_model(imag_modelstates)
            imag_value = imag_value_dist.mean
            discount_dist = self.world_model.discount_model(imag_modelstates)
            discount_arr = self.world_model.discount*torch.round(discount_dist.base_dist.probs)              #mean = prob(disc==1)

        actor_loss, discount, lambda_returns = self.actor_loss(imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy)
        value_loss = self.value_loss(imag_modelstates, discount, lambda_returns)     

        # mean_target = torch.mean(lambda_returns, dim=1)
        # max_targ = torch.max(mean_target).item()
        # min_targ = torch.min(mean_target).item() 
        # std_targ = torch.std(mean_target).item()
        # mean_targ = torch.mean(mean_target).item()

        return actor_loss, value_loss


def compute_return(
            reward: torch.Tensor,
            value: torch.Tensor,
            discount: torch.Tensor,
            bootstrap: torch.Tensor,
            lambda_: float
        ):
    """
    Compute the discounted reward for a batch of data.
    reward, value, and discount are all shape [horizon - 1, batch, 1] (last element is cut off)
    Bootstrap is [batch, 1]
    """
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    target = reward + discount * next_values * (1 - lambda_)
    timesteps = list(range(reward.shape[0] - 1, -1, -1))
    outputs = []
    accumulated_reward = bootstrap
    for t in timesteps:
        inp = target[t]
        discount_factor = discount[t]
        accumulated_reward = inp + discount_factor * lambda_ * accumulated_reward
        outputs.append(accumulated_reward)
    returns = torch.flip(torch.stack(outputs), [0])
    return returns