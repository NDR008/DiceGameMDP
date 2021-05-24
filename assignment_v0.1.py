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
        # import time
        # start_time = time.process_time()
        super().__init__(game)
        self.gamma = 0.7
        self.theta = 0.01
        poss_act = self.game.actions
        poss_states = self.game.states
        self.vals0 = {}
        for each_state in poss_states:
            self.vals0[(each_state)] = [0, None]

        zeroed_acts = {}
        for act in poss_act:
            zeroed_acts[(act)] = 0


        while 1:
            delta = 0
            for each_state in poss_states:
                temp = self.vals0[each_state][0]
                acts = zeroed_acts.copy()
                for act in poss_act:
                    (state_dashes, flag, reward, probability)  = self.game.get_next_states(act, each_state)
                    for prob_pos, each_probability in enumerate(probability):
                        if not flag:
                            state_dash = state_dashes[prob_pos]
                            acts[act] += each_probability * (reward + self.gamma * self.vals0[state_dash][0])
                        else:
                            acts[act] += each_probability * reward
                best_action_val = max(acts.values())
                best_action = max(acts, key=acts.get)
                self.vals0[each_state] = [best_action_val, best_action]
            delta = max(delta, abs(temp - self.vals0[each_state][0]))
            if delta < self.theta:
                break
        # end_time = time.process_time()
        # print(end_time-start_time)

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

    np.random.seed(1)
    #game = DiceGame(3,6,None,None,5)
    game = DiceGame()
    agent2 = MyAgent(game)
    b = []
    a=[5,50,500,5000,50000]
    for cycle in a:
        for i in range(cycle):
            # agent1 = AlwaysHold(game)
            # play_game_with_agent(agent1, game, verbose=True)
            b.append(play_game_with_agent(agent2, game, verbose=False))

        print("in ", cycle, " cycles... min score:", min(b), "max score:", max(b), "average:", np.average(b))
if __name__ == "__main__":
    main()