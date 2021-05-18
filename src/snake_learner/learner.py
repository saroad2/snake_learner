import json
from collections import defaultdict

import numpy as np

from snake_learner.board import SnakeBoard
from snake_learner.direction import Direction


class SnakeLearner:

    def __init__(
        self,
        rows,
        columns,
        view_getter,
        discount_factor,
        alpha,
        epsilon,
        loss_change,
        reward_change,
        distance_change,
        loss_penalty,
        eat_reward,
    ):
        self.rows = rows
        self.columns = columns
        self.view_getter = view_getter
        self.q = defaultdict(lambda: np.zeros(len(Direction)))

        self.discount_factor = discount_factor
        self.alpha = alpha
        self.epsilon = epsilon
        self.loss_change = loss_change
        self.reward_change = reward_change
        self.distance_change = distance_change
        self.eat_reward = eat_reward
        self.loss_penalty = loss_penalty

        self.history = []

    @property
    def max_score(self):
        return np.max([history_point["score"] for history_point in self.history])

    @property
    def max_rewards_sum(self):
        return np.max([history_point["rewards_sum"] for history_point in self.history])

    @property
    def longest_duration(self):
        return np.max([history_point["duration"] for history_point in self.history])

    def load_q_from_file(self, q_file_path):
        with open(q_file_path, mode="r") as fd:
            new_q = json.load(fd)
        self.q.update({key: np.array(val) for key, val in new_q.items()})

    def run_train_iteration(self):
        board = SnakeBoard(rows=self.rows, columns=self.columns)
        iterations = 0
        rewards_list = []
        while True:
            iterations += 1

            reward = self.make_move(board)

            rewards_list.append(reward)

            # done is True if episode terminated
            if board.done:
                break
        self.history.append(
            dict(
                score=board.score,
                duration=iterations,
                rewards_sum=np.sum(rewards_list),
                rewards_max=np.max(rewards_list),
                states=len(self.q)
            )
        )

    def make_move(self, board, update_q=True):
        # get probabilities of all actions from current state
        state = self.view_getter.get_view(board)
        action_probabilities = self.get_policy(state)
        # choose action according to
        # the probability distribution
        action_index = np.random.choice(
            np.arange(len(Direction)),
            p=action_probabilities,
        )
        # take action and get reward, transit to next state
        reward = self.run_step(
            board=board, direction=Direction(action_index)
        )

        if update_q:
            td_target = reward + self.discount_factor * self.best_reward(board)
            td_delta = td_target - self.q[state][action_index]
            self.q[state][action_index] += self.alpha * td_delta
        return reward

    def get_policy(self, state):
        action_probabilities = np.ones(len(Direction),
                                       dtype=float) * self.epsilon / len(Direction)

        best_action = np.argmax(self.q[state])
        action_probabilities[best_action] += (1.0 - self.epsilon)
        return action_probabilities

    def run_step(self, board, direction):
        initial_score = board.score

        board.move(direction)

        new_score = board.score
        if new_score > initial_score:
            return self.eat_reward * np.exp(self.reward_change * new_score)
        if board.done:
            return -self.loss_penalty * np.exp(self.loss_change * initial_score)
        food_direction = board.food - board.head
        food_distance = int(np.sum(np.fabs(food_direction)))
        return np.exp(-self.distance_change * food_distance)

    def best_reward(self, board):
        if board.done:
            return 0
        state = self.view_getter.get_view(board)
        best_next_action = np.argmax(self.q[state])
        return self.q[state][best_next_action]
