from collections import defaultdict

import numpy as np

from snake_learner.board import SnakeBoard
from snake_learner.direction import Direction
from snake_learner.view_getter import ViewGetter


class SnakeLearner:

    ACTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

    def __init__(
        self,
        rows,
        columns,
        sight_distance,
        discount_factor=1,
        alpha=0.6,
        epsilon=0.1,
        reward_decay=0.15,
        loss_penalty=0,
        eat_reward=10,
    ):
        self.rows = rows
        self.columns = columns
        self.view_getter = ViewGetter(sight_distance=sight_distance)
        self.q = defaultdict(lambda: np.zeros(self.num_actions))

        self.discount_factor = discount_factor
        self.alpha = alpha
        self.epsilon = epsilon
        self.reward_decay = reward_decay
        self.eat_reward = eat_reward
        self.loss_penalty = loss_penalty

        self.history = []

    @property
    def num_actions(self):
        return len(self.ACTIONS)

    @property
    def max_score(self):
        return np.max([history_point["score"] for history_point in self.history])

    @property
    def max_rewards_sum(self):
        return np.max([history_point["rewards_sum"] for history_point in self.history])

    @property
    def longest_duration(self):
        return np.max([history_point["duration"] for history_point in self.history])

    def run_iteration(self):
        board = SnakeBoard(rows=self.rows, columns=self.columns)
        iterations = 0
        last_score_up = 0
        rewards_list = []
        while True:
            iterations += 1

            # get probabilities of all actions from current state
            state = self.view_getter.get_view(board)
            action_probabilities = self.get_policy(state)

            # choose action according to
            # the probability distribution
            action_index = np.random.choice(
                np.arange(self.num_actions),
                p=action_probabilities,
            )

            # take action and get reward, transit to next state
            reward, last_score_up = self.run_step(
                board=board,
                direction=self.ACTIONS[action_index],
                iteration=iterations,
                last_score_up=last_score_up
            )
            rewards_list.append(reward)

            # TD Update
            next_state = self.view_getter.get_view(board)
            best_next_action = np.argmax(self.q[next_state])
            td_target = reward + self.discount_factor * self.q[next_state][best_next_action]
            td_delta = td_target - self.q[state][action_index]
            self.q[state][action_index] += self.alpha * td_delta

            # done is True if episode terminated
            if board.done:
                break
        self.history.append(
            dict(
                score=board.score,
                duration=iterations,
                rewards_sum=np.sum(rewards_list),
                rewards_max=np.max(rewards_list),
            )
        )

    def get_policy(self, state):
        action_probabilities = np.ones(self.num_actions,
                                       dtype=float) * self.epsilon / self.num_actions

        best_action = np.argmax(self.q[state])
        action_probabilities[best_action] += (1.0 - self.epsilon)
        return action_probabilities

    def run_step(self, board, direction, iteration, last_score_up):
        initial_score = board.score

        board.move(direction)

        new_score = board.score
        if new_score > initial_score:
            return self.eat_reward * np.exp(self.reward_decay * new_score), iteration
        if board.done:
            return -self.loss_penalty, last_score_up
        return np.exp(-self.reward_decay * (iteration - last_score_up)), last_score_up
