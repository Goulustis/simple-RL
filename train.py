from environment import Game
import matplotlib.pyplot as plt
from model import Agent
import numpy as np

def plotLearning(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


if __name__ == '__main__':
    env = Game() #gym.make('LunarLander-v2')
    agent1 = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01,
                  input_dims=[8], lr=0.001, player_code = 1)
    agent2 = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01,
                  input_dims=[8], lr=0.001, player_code = 2)
    
    up_targ_iter = 50
    scores1, eps_history1 = [], []
    scores2, eps_history2 = [], []
    n_games = 50000
    
    for i in range(n_games):
        score1 = 0
        score2 = 0
        done = False
        observation = env.reset()
        p1_hist, p2_hist = observation
        while not done:
            action1 = agent1.choose_action(p2_hist)
            action2 = agent2.choose_action(p1_hist)

            observation_, reward, done, info = env.step(action1, action2)

            rw1, rw2 = reward
            score1 += rw1
            score2 += rw2 

            p1_hist_, p2_hist_ = observation_

            agent1.store_transition(p2_hist, action1, rw1, 
                                    p2_hist_, done)

            agent2.store_transition(p1_hist, action2, rw2,
                                    p1_hist_, done)
            agent1.learn()
            agent2.learn()
            p1_hist, p2_hist = observation_

        if i%up_targ_iter == 0 and i > 0:
            agent1.update_targ_model()
            agent2.update_targ_model()

        scores1.append(score1)
        eps_history1.append(agent1.epsilon)

        scores2.append(score1)
        eps_history2.append(agent1.epsilon)

        avg_score = np.mean(scores1[-100:])

        print('episode ', i, 'score1 %.2f' % score1,
                'average score1 %.2f' % avg_score,
                'epsilon1 %.2f' % agent1.epsilon)
        
        print('episode ', i, 'score2 %.2f' % score2,
                'average score2 %.2f' % avg_score,
                'epsilon2 %.2f' % agent2.epsilon)
    x = [i+1 for i in range(n_games)]
    plotLearning(x, scores1, eps_history1, "p1.png")
    plotLearning(x, scores2, eps_history1, "p2.png")