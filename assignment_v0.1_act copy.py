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
        debug = False

        import time
        start_time = time.process_time()
        super().__init__(game)
        self.gamma = gamma
        self.theta = theta

        poss_act = self.game.actions
        #choices = len(poss_act)
        poss_states = self.game.states
        self.vals0 = {}
        for each_state in poss_states:
            #self.vals0[(each_state)] = [0, poss_act[np.random.randint(choices)]]
            self.vals0[(each_state)] = ([0, poss_act[-1]])
        
        run_next_states = {}
        for each_state in poss_states:
            for act in poss_act:
                run_next_states[(act, each_state)] = self.game.get_next_states(act, each_state)

        pre_load = time.process_time() - start_time
        i = 0
        for extra in range(1):  # just for the rare possibility that the values lack of convergence will cause an issue
            while 1:
                i += 1
                convergence = 1  # hopefully
                for each_state in poss_states:
                    old_act = self.vals0[each_state][1]
                    max_val = 0
                    max_act = 0
                    for act in poss_act:
                        new_val = 0
                        (state_dashes, flag, reward, probability)  = run_next_states[(act, each_state)]
                        for prob_pos, each_probability in enumerate(probability):
                            if not flag:
                                state_dash = state_dashes[prob_pos]
                                new_val += each_probability * (reward + self.gamma * self.vals0[state_dash][0])
                            else:
                                new_val += each_probability * (reward)
                        if new_val > max_val:
                            max_val = new_val
                            max_act = act        # this is the hybrid stage of argmax
                    self.vals0[each_state] = [max_val, max_act]
                    if self.vals0[each_state][1] != old_act:
                        convergence = 0  # need to try again
                delta_time = time.process_time() - start_time
                response_time = (delta_time-pre_load) / i
                next_loop_time = response_time + delta_time
                if convergence or next_loop_time > 30:
                    break
            if debug: print("Policy based init time",pre_load, time.process_time()-start_time-pre_load, i)
            self.time = time.process_time() - start_time
            self.loops = i
        
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

    n = [100000]
    thetas = [0.001]
    gammas = [1]
    results = []
    penalties = [0.5,0,1]
    for cycle in n:
        for val in penalties:
            for theta in thetas:
                for gamma in gammas:


                    np.random.seed(1)
                    #game = DiceGame(dice=3, sides=6, bias=[0.1, 0.1, 0.1, 0.5, 0.1, 0.1], penalty = val)
                    #game =  DiceGame(dice=2, sides=3, values=[1, 2, 6], bias=[0.5, 0.1, 0.4], penalty=2)
                    game = DiceGame(dice=6)
                    agent2 = MyAgent(game, theta, gamma)
                    init_time = agent2.time
                    init_loops = agent2.loops

                    for i in range(cycle):
                        results.append(play_game_with_agent(agent2, game, verbose=False))
                    print("time:", init_time,  "\t", "loop:", init_loops, "cycle:",cycle,  "\t", "theta:", theta, "\t", "gamma:", gamma,  "\t", "penalty:", val, "\t", "min:", min(results),  "\t", "max:", max(results), "\t", "average", np.average(results))
                    import matplotlib.pyplot as plt
                    plt.hist(results, bins = range(-20,30))
                    title  = "Value iter Game_results (2 dice)" + str(theta) + " - " + str(gamma) + " - " + str(val)
                    file = title + ".png"
                    plt.title(title)
                    plt.savefig(file)
                    plt.clf()
                    results = []
if __name__ == "__main__":
    main()