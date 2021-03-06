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
            self.vals0[(each_state)] = [0, None]
        # value iteration

        poss_act = self.game.actions
        zeroed_acts = {}
        for act in poss_act:
            zeroed_acts[(act)] = 0

        loop = 0
        for i in range(1):
            loop += 1
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
                            # acts[act] += each_probability * (reward + 0 * self.vals0[state_dash][0])
                            acts[act] += each_probability * (reward)
                best_action_val = max(acts.values())
                best_action = max(acts, key=acts.get)
                self.vals0[each_state] = [best_action_val, best_action]
                delta = max(delta, abs(temp - self.vals0[each_state][0]))
            delta_time = time.process_time()-start_time
            if delta < self.theta or delta_time > 25:
                break
        if debug: print(delta, time.process_time()-start_time)

        # policy improvement after a single pass of value iteration
        while 1:
            policy_stable = True
            for each_state in poss_states:
                temp = self.vals0[each_state][1]
                acts = zeroed_acts.copy()
                for act in poss_act:
                    (state_dashes, flag, reward, probability)  = self.game.get_next_states(act, each_state)
                    for prob_pos, each_probability in enumerate(probability):
                        if not flag:
                            state_dash = state_dashes[prob_pos]
                            acts[act] += each_probability * (reward + self.gamma * self.vals0[state_dash][0])
                        else:
                            # acts[act] += each_probability * (reward + 0 * self.vals0[state_dash][0])
                            acts[act] += each_probability * (reward)
                best_action_val = max(acts.values())
                best_action = max(acts, key=acts.get)
                self.vals0[each_state] = [best_action_val, best_action]
                if temp != self.vals0[each_state][1]:
                    policy_stable = False
                    #print(temp, self.vals0[each_state][1])
            if policy_stable:
                break


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
    # ??or delete the line to make it change each time it is run

    #a=[10,100,10000]
    a=[10000]
    b = []
    for cycle in a:
        for theta in [0.001, 0.00001, 0.0000000001]:
            for gamma in [1.0]:

                np.random.seed(1)
                #game = DiceGame()
                game = DiceGame(3, 6, None, None, 1)
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