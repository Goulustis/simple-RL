import numpy as np

class Game:
    def __init__(self, game_lim=10000, act_hist_lim = 256):
        self.p1_score = 0
        self.p2_score = 0
        
        self.p1_mtx = np.array([[4, 0],
                                [5, 2]])

        self.p2_mtx = np.array([[4, 5],
                                [0, 2]])


        self.p1_act_hist = []
        self.p2_act_hist = []

        self.game_lim = game_lim
        self.act_hist_lim = act_hist_lim
        self.curr_iter = 0

    def reset(self):
        self.p1_score = 0
        self.p2_score = 0
        self.curr_iter = 0
        self.p1_act_hist = []
        self.p2_act_hist = []

        new_obs = (self.p1_act_hist, self.p2_act_hist)
        return new_obs
    

    def step(self, p1_act, p2_act):
        self.curr_iter += 1 

        p1_rw = self.p1_mtx[p1_act, p2_act]
        p2_rw = self.p2_mtx[p1_act, p2_act]

        self.p1_act_hist.append(p1_act)
        self.p2_act_hist.append(p2_act)

        if len(self.p1_act_hist) > self.act_hist_lim:
            self.p1_act_hist.pop(0)
            self.p2_act_hist.pop(0)

        done = self.curr_iter > self.game_lim
        new_obs = (self.p1_act_hist, self.p2_act_hist)
        return new_obs, (p1_rw, p2_rw), done