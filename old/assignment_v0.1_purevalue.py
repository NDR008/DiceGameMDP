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
    def __init__(self, game, theta = 0.01, gamma = 0.9):  # to remove
    # def __init__(self, game):        
        super().__init__(game)
        self.gamma = gamma
        #self.theta = theta
        
        debug = True  # to remove
        import time  # to remove
        start_time = time.process_time()  # to remove

        poss_act = self.game.actions
        poss_states = self.game.states

        derived_states = {}
        for each_state in poss_states:
            self.policy[(each_state)] = [0, poss_act[-1]]
            for act in poss_act:
                derived_states[(act, each_state)] = self.game.get_next_states(act, each_state)
        i = 0
        while 1:
            i += 1
            delta = 0
            for each_state in poss_states:
                old_val = self.policy[each_state][0]
                max_act = self.policy[each_state][1]
                max_val = old_val
                for act in poss_act:
                    new_val = 0
                    (state_dashes, flag, reward, probability)  = derived_states[(act, each_state)]
                    for prob_pos, each_probability in enumerate(probability):
                        if not flag:
                            state_dash = state_dashes[prob_pos]
                            new_val += each_probability * (reward + self.gamma * self.policy[state_dash][0])
                        else:
                            new_val += each_probability * (reward)
                    if new_val > max_val:
                        max_val = new_val
                        max_act = act        # this is the hybrid stage of argmax
                self.policy[each_state] = [max_val, max_act]
                delta = max(delta, abs(old_val - self.policy[each_state][0]))
            if delta < self.theta:
                break
        if debug: print("Policy based init time",pre_load, time.process_time()-start_time-pre_load, i)
        print(self.policy)

        

    def play(self, state):
        # maybe if we are in the winning state, we can skip evaluating
        return self.policy[state][1]
    
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
    thetas = [0.001]
    #thetas = [0.1]
    b = []
    for cycle in a:
        for theta in thetas:
            for gamma in [1]:

                np.random.seed(1)
                
                #game = DiceGame(dice=3, bias = [0.3, 0.05, 0.05, 0.05, 0.05, 0.3, 0.05, 0.05 ,0.05, 0.05], sides=10, penalty=0)
                #game =  DiceGame(dice=2, sides=3, values=[1, 2, 6], bias=[0.5, 0.1, 0.4], penalty=2)
                #game = DiceGame(sides=4, dice=2, values=[-50, -25, -2, 5], bias = [0.1, 0.1, 0.1, 0.7])
                #game = DiceGame(sides=4, dice=2, values=[-50, -25, -2, 5], bias = [0.1, 0.1, 0.1, 0.7], penalty=0)
                #game = DiceGame()
                game = DiceGame(sides=4, dice=2, values=[-50, -25, -5, -2], bias = [0.1, 0.1, 0.1, 0.7], penalty=0)
                agent2 = MyAgent(game, theta, gamma)

                for i in range(cycle):
                    # agent1 = AlwaysHold(game)
                    # play_game_with_agent(agent1, game, verbose=True)
                    b.append(play_game_with_agent(agent2, game, verbose=False))
                print(theta, "\t", gamma,  "\t", cycle,  "\t", min(b),  "\t", max(b),  "\t", np.average(b))
                import matplotlib.pyplot as plt
                plt.hist(b, bins = range(-20,20))
                title  = "Value-iter: Game_results " + str(theta) + " - " + str(gamma)
                file = title + ".png"
                plt.title(title)
                plt.savefig(file)
                plt.clf()
                b = []
if __name__ == "__main__":
    main()