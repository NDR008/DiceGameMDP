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
    def __init__(self, game): 
        super().__init__(game)
        self.gamma = 1
        
        possible_action = self.game.actions
        possible_states = self.game.states
        self.policy = {}
        derived_states = {}
        
        # initialise the policy
        for each_state in possible_states:
            self.policy[each_state] = [0, possible_action[-1]]
            for action in possible_action:
                derived_states[(action, each_state)] = self.game.get_next_states(action, each_state)

        while 1:
            convergence = 1  # hopefully
            for each_state in possible_states:
                max_val, old_act = self.policy[each_state]
                max_act = old_act  # in case no new value exceed the old
                
                for action in possible_action:
                    new_val = 0
                    (state_dashes, flag, reward, probability)  = derived_states[(action, each_state)]
                    for index, each_probability in enumerate(probability):
                        if not flag:
                            state_dash = state_dashes[index]
                            new_val += each_probability * (reward + self.gamma * self.policy[state_dash][0])
                        else:
                            new_val += each_probability * (reward)
                    if new_val > max_val:
                        max_val = new_val
                        max_act = action        # this is the hybrid stage of argmax
                self.policy[each_state] = [max_val, max_act]  # put back same val or new one
                if self.policy[each_state][1] != old_act:
                    convergence = 0  # need to try again
            if convergence:
                break

    def play(self, state):
        return self.policy[state][1]

##############################################################
        
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

    if(verbose): print(f"Start state: {start_state}, \t{type(agent).__name__}, \tFinal dice: {state}, score: {game.score}") 
        
    return game.score


def main():
    # random seed makes the results deterministic
    # change the number to see different results
    #  or delete the line to make it change each time it is run
    from scipy.stats import skew
    
    a=[100, 10000, 1000000]
    #a=[10]

    games = [DiceGame(),
             DiceGame(dice=5, sides=8),
             DiceGame(dice=6, sides=5),
             DiceGame(dice=3, sides=6, penalty=5),
             DiceGame(dice=2, sides=4, values=[-50, -25, -2, 5], bias = [0.1, 0.1, 0.1, 0.7])]
    gammas=[1.0]
    #gammas=[-1]
    
    print("game", "\t", "gamma",  "\t", "cycle",  "\t", "min",  "\t", "max",  "\t", "avgerage  ", "\t", "time   ", "\t", "loops", "\t", "skew")          
    for game_id, game in enumerate(games):
        for cycle in a:
                for gamma in gammas:
                    result=[]
                    np.random.seed(1)
                    #game = DiceGame()
                    #game = DiceGame(sides=4, dice=2, values=[-50, -25, -5, -2], bias = [0.1, 0.1, 0.1, 0.7], penalty=0)
                    agent2 = MyAgent(game)

                    
                    for i in range(cycle):
                        result.append(play_game_with_agent(agent2, game, verbose=False))
                    print(game_id, "\t",gamma, "\t", cycle,  "\t", min(result),  "\t", max(result),  "\t", np.average(result), "\t", skew(result))
                    
                    #for state in game.states:
                        #print(agent2.policy)
                    
                    import matplotlib.pyplot as plt
                    plt.grid(axis='y', alpha=0.75)
                    plt.hist(result, bins=(max(result)-min(result)), alpha=0.7, rwidth=0.85)
                    title  = "Sanity_check" + str(game_id) + "-" + str(gamma) + "-" + str(cycle)
                    file = "images/" + title + ".png"
                    plt.title(title)
                    plt.savefig(file)
                    plt.clf()
if __name__ == "__main__":
    main()