import json
from collections import defaultdict

import numpy as np

from snake_learner.board import SnakeBoard
from snake_learner.snake_action import SnakeAction
from snake_learner.linalg_util import block_distance


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
        move_reward,
        max_moves_to_score=None,
    ):
        self.rows = rows
        self.columns = columns
        self.max_moves_to_score = max_moves_to_score
        self.view_getter = view_getter
        self.q = defaultdict(lambda: np.zeros(len(SnakeAction)))

        self.discount_factor = discount_factor
        self.alpha = alpha
        self.epsilon = epsilon
        self.loss_change = loss_change
        self.reward_change = reward_change
        self.distance_change = distance_change
        self.eat_reward = eat_reward
        self.loss_penalty = loss_penalty
        self.move_reward = move_reward

        self.history = []
        self.best_game = None

    @property
    def max_score(self):
        return self.field_max("score")

    @property
    def max_rewards(self):
        return self.field_max("rewards")

    @property
    def longest_duration(self):
        return self.field_max("duration")

    @property
    def states_number(self):
        return len(self.q)

    def recent_scores_mean(self, n=1_000):
        return self.recent_field_mean(field="score", n=n)

    def recent_rewards_mean(self, n=1_000):
        return self.recent_field_mean(field="rewards", n=n)

    def recent_duration_mean(self, n=1_000):
        return self.recent_field_mean(field="duration", n=n)

    def field_max(self, field):
        if len(self.history) == 0:
            return None
        return np.max([history_point[field] for history_point in self.history])

    def recent_field_mean(self, field, n):
        if len(self.history) == 0:
            return None
        history = self.history
        if len(history) > n:
            history = history[-n:]
        return np.mean([history_point[field] for history_point in history])

    def load_q_from_file(self, q_file_path):
        with open(q_file_path, mode="r") as fd:
            new_q = json.load(fd)
        self.q.update({key: np.array(val) for key, val in new_q.items()})

    def build_board(self):
        return SnakeBoard(
            rows=self.rows,
            columns=self.columns,
            max_moves_to_score=self.max_moves_to_score,
        )

    def run_train_iteration(self):
        board = self.build_board()
        rewards = 0
        while True:
            reward = self.make_move(board)
            rewards += reward

            # done is True if episode terminated
            if board.done:
                break
        self.history.append(
            dict(
                score=board.score,
                duration=board.moves,
                rewards=rewards,
                states=self.states_number,
                velocity=board.score / board.moves,
            )
        )
        if self.best_game is None or board.score > self.best_game.score:
            self.best_game = board

    def play(self):
        board = self.build_board()
        while not board.done:
            self.make_move(board)
        return board

    def make_move(self, board, update_q=True):
        # get probabilities of all actions from current state
        state = self.view_getter.get_view(board)
        action_probabilities = self.get_policy(state)
        # choose action according to
        # the probability distribution
        action_index = np.random.choice(
            np.arange(len(SnakeAction)),
            p=action_probabilities,
        )
        # take action and get reward, transit to next state
        reward = self.run_step(
            board=board, action=SnakeAction(action_index)
        )

        if update_q:
            td_target = reward + self.discount_factor * self.best_reward(board)
            td_delta = td_target - self.q[state][action_index]
            self.q[state][action_index] += self.alpha * td_delta
        return reward

    def get_policy(self, state):
        action_probabilities = np.ones(
            len(SnakeAction), dtype=float
        ) * self.epsilon / len(SnakeAction)

        best_action = np.argmax(self.q[state])
        action_probabilities[best_action] += (1.0 - self.epsilon)
        return action_probabilities

    def run_step(self, board, action):
        initial_score = board.score

        board.turn(action)

        new_score = board.score
        if new_score > initial_score:
            return self.eat_reward * np.exp(self.reward_change * new_score)
        if board.lost:
            return -self.loss_penalty * np.exp(self.loss_change * initial_score)
        food_direction = board.food - board.head
        food_distance = block_distance(food_direction)
        return self.move_reward * np.exp(-self.distance_change * food_distance)

    def best_reward(self, board):
        if board.done:
            return 0
        state = self.view_getter.get_view(board)
        best_next_action = np.argmax(self.q[state])
        return self.q[state][best_next_action]
