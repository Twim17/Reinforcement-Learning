from re import A
from env import *
import torch
import argparse
from obs import *
from rssm import *
from actor_critic import *
from action_model import *
from world_model import *
from trainer import *

def main():
    parser = argparse.ArgumentParser(description='Run tests.')
    parser.add_argument('-g','--game_type', type=str, default='minatari')
    parser.add_argument('-r','--render', type=str, default='rgb_array')
    parser.add_argument('-env','--env_name', type=str, default='Breakout-v0')
    parser.add_argument('-bs','--batch_size', type=int, default=50)
    parser.add_argument('-sl','--seq_len', type=int, default=50)
    parser.add_argument('-rt','--resume_training', type=int, default=0)
    parser.add_argument('-ev','--eval', type=int, default=0)
    parser.add_argument('-ar','--action_repeat', type=int, default=1)


    args = parser.parse_args()
    game_type = args.game_type
    render_mode = args.render
    env_name = args.env_name
    batch_size = args.batch_size
    seq_len = args.seq_len
    resume_training = args.resume_training
    is_eval = args.eval
    action_repeat = args.action_repeat
    model_dir = 'saved_models/'


    np.random.seed(69420777)
    torch.manual_seed(69420777)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    #device = torch.device('cpu')
    env = None
    if game_type == 'minatari':
        # env_name = args.env_name.lower()[:-3]
        env = GymMinAtar(env_name, action_repeat=action_repeat)
    else:
        env = GymAtari(env_name, action_repeat=action_repeat, render_mode=render_mode)
    print(game_type)
    #obs = env.reset()
    action_size = env.act_shape
    print("action_size",action_size)
    trainer = Trainer(env_name, action_size, game_type=game_type, device=device)

    #COLLECT EXPERIENCE FOR BUFFER
    if resume_training > 0:
        #saved_dict = torch.load(args.path)
        #trainer.load_save_dict(saved_dict)
        print('resuming training for step: ', resume_training)
    
    if is_eval != 0:
        eval_dir = model_dir + env_name + '/' + 'models_' + str(resume_training) + '.pth'
        trainer.evaluation(env, eval_dir, is_eval)
    else:
        print('...training...')
        trainer.train(env, resume_training)



if __name__ == '__main__':
    main()