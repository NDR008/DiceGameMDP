from abc import ABC, abstractmethod
from dice_game import DiceGame
import numpy as np



class DiceGameAgent(ABC):
    def __init__(self, game):
        self.game = game
    
    @abstractmethod
    def play(self, state):
        pass


class AlwaysHold(DiceGameAgent):
    def play(self, state):
        return (0, 1, 2)


class PerfectionistAgent(DiceGameAgent):
    def play(self, state):
        if state == (1, 1, 1) or state == (1, 1, 6):
            return (0, 1, 2)
        else:
            return ()

###############################################################

class MyAgent(DiceGameAgent):
    def __init__(self, game, theta = 0.01, gamma = 0.9):
        debug = True

        import time
        start_time = time.process_time()
        super().__init__(game)
        self.gamma = gamma
        self.theta = theta

        poss_states = self.game.states
        self.vals0 = {}
        for each_state in poss_states:
            self.vals0[(each_state)] = [0, None, False]
        # value iteration

        poss_act = self.game.actions
        
        run_next_states = {}
        for each_state in poss_states:
            for act in poss_act:
                run_next_states[(act, each_state)] = self.game.get_next_states(act, each_state)
        
        finished = False
        while not finished:
            delta = 0
            finished = True
            for each_state in poss_states:
                #acts = zeroed_acts.copy()
                max_val = 0
                max_act = 0
                if self.vals0[each_state][2]:
                    continue
                temp = self.vals0[each_state][0]
                for act in poss_act:
                    accu = 0
                    (state_dashes, flag, reward, probability)  = run_next_states[(act, each_state)]
                    for prob_pos, each_probability in enumerate(probability):
                        if not flag:
                            state_dash = state_dashes[prob_pos]
                            accu += each_probability * (reward + self.gamma * self.vals0[state_dash][0])
                        else:
                            accu += each_probability * (reward)
                    if accu > max_val:
                        max_val = accu
                        max_act = act
                self.vals0[each_state] = [max_val, max_act]
                delta = max(delta, abs(temp - self.vals0[each_state][0]))
                if delta < self.theta:
                    self.vals0[each_state] = [max_val, max_act, True]
                else:
                    finished = False 
                    self.vals0[each_state] = [max_val, max_act, False]
                print("state is", self.vals0[each_state][2])
            delta_time = time.process_time()-start_time
            if finished:
                break
        if debug: print("delta & init time", delta, time.process_time()-start_time)
        
        

    def play(self, state):
        # maybe if we are in the winning state, we can skip evaluating
        return self.vals0[state][1]
    
###############################################################
        
def play_game_with_agent(agent, game, verbose=False):
    state = game.reset()
    
    if(verbose): print(f"Testing agent: \n\t{type(agent).__name__}")
    if(verbose): print(f"Starting dice: \n\t{state}\n")
    start_state = state
        
    game_over = False
    actions = 0
    while not game_over:
        action = agent.play(state)  # make a move based on the state
        actions += 1  # count how many moves
        
        if(verbose): print(f"Action {actions}: \t{action}")
        _, state, game_over = game.roll(action)
        if(verbose and not game_over): print(f"Dice: \t\t{state}")

    #if(verbose): print(f"\nFinal dice: {state}, score: {game.score}")
    if(verbose): print(f"Start state: {start_state}, \t{type(agent).__name__}, \tFinal dice: {state}, score: {game.score}")
        
    return game.score


def main():
    # random seed makes the results deterministic
    # change the number to see different results
    #  or delete the line to make it change each time it is run

    #a=[10,100,10000]
    a=[10000]
    thetas = [0.01, 0.00001, 0.0000000001]
    thetas = [55]
    b = []
    for cycle in a:
        for theta in thetas:
            for gamma in [1.0]:

                np.random.seed(1)
                #game = DiceGame()
                game = DiceGame(dice=5)
                agent2 = MyAgent(game, theta, gamma)

                for i in range(cycle):
                    # agent1 = AlwaysHold(game)
                    # play_game_with_agent(agent1, game, verbose=True)
                    b.append(play_game_with_agent(agent2, game, verbose=False))
                print(theta, "\t", gamma,  "\t", cycle,  "\t", min(b),  "\t", max(b),  "\t", np.average(b))
                import matplotlib.pyplot as plt
                plt.hist(b, bins = range(-20,20))
                title  = "Game_results " + str(theta) + " - " + str(gamma)
                file = title + ".png"
                plt.title(title)
                plt.savefig(file)
                plt.clf()
                b = []
if __name__ == "__main__":
    main()