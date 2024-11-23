from typing import Tuple

import numpy as np
from gymnasium import Env, spaces

X = ord('X')
O = ord('O')
EMPTY = ord('•')

class BoardEnv(Env):
    """Среда - поле для игры в крестики-нолики, на котором играют агенты"""

    def __init__(self, size: int = 3):
        """

        :param size: сторона квадратного поля
        """

        self.size = size
        self.board = np.full((size, size), EMPTY, dtype=int)
        self.current_player = X
        self.current_turn = 0
        self.row, self.col = -1, -1

        self.action_space = spaces.Discrete(size * size, start=1)
        self.observation_space = spaces.MultiDiscrete([X, O, EMPTY] * (size * size))

        self.reset()

    def _get_obs(self) -> np.ndarray:
        """Преобразуем состояние среды (environment) в наблюдение (observation).
        Эти данные подаются на обучение нейронной сети

        :return: массив-наблюдение
        """

        board = list(self.board.flatten())
        current_player = X if self.current_player == O else O

        return np.array(board + [current_player])

    def _get_info(self) -> dict:
        """Диагностические данные, которые позволяют отслеживать работу нейронной сети

        :return: массив диагностической информации
        """
        vectorized_chr = np.vectorize(chr)
        human_readable_board = vectorized_chr(self.board)

        human_readable_current_player = chr(self.current_player)

        return {
            'board': human_readable_board,
            'current_player': human_readable_current_player,
            'current_turn': self.current_turn
        }

    def print_diagnostic(self, _get_info_dict: dict, terminated: bool):

        print(f'Ход #{_get_info_dict['current_turn'] + 1}', end=', ')
        print('ходит', end=' ')
        if terminated:
            print('первый игрок' if _get_info_dict['current_player'] == 'X' else 'второй игрок,', end=' ')
        else:
            print('первый игрок' if _get_info_dict['current_player'] == 'O' else 'второй игрок,', end=' ')
        print(f'ход: {(int(self.row), int(self.col))}')

        print('Поле после данного хода: ')
        human_readable_board = _get_info_dict['board']

        for row in human_readable_board:
            for symbol in row:
                print(symbol, end=' ')
            print('')

        print('Победа!' if terminated else '')

    def reset(self, seed=None, options=None):
        """Инициализирует новый эпизод для среды.

        :param seed: Инициализирует генератор случайных чисел для детерминированного состояния.
        :param options: Дополнительные параметры для настройки сброса.
        :return: Кортеж из начального наблюдения и дополнительной информации.
        """

        super().reset(seed=seed)

        self.board = np.full((self.size, self.size), EMPTY, dtype=int)
        self.current_player = X
        self.current_turn = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def check_winner(self, player):
        """Проверяет, выиграл ли игрок (присутствуют ли на поле три символа в ряд)

        :param player: переменная, обозначающая игрока
        :return: True, если игрок выиграл, False в противном случае
        """

        rows, cols = self.board.shape

        # Проверка строк
        for row in range(rows):
            for col in range(cols - 2):
                if all(self.board[row, col:col + 3] == player):
                    return True

        # Проверка столбцов
        for row in range(rows - 2):
            for col in range(cols):
                if all(self.board[row:row + 3, col] == player):
                    return True

        # Проверка диагоналей (сверху вниз, слева направо)
        for row in range(rows - 2):
            for col in range(cols - 2):
                if all(self.board[row + k, col + k] == player for k in range(3)):
                    return True

        # Проверка диагоналей (сверху вниз, справа налево)
        for row in range(rows - 2):
            for col in range(2, cols):
                if all(self.board[row + k, col - k] == player for k in range(3)):
                    return True

        return False

    def step(self, action: int) -> Tuple[np.ndarray, int, bool, bool, dict]:
        """Выполняет действие, вычисляет новое состояние среды и возвращает кортеж из следующего наблюдения,
        вознаграждения, признаков завершения и дополнительной информации.

        :param action: Действие, выполняемое агентом.
        :return: Кортеж из следующего наблюдения, вознаграждения, признаков завершения и дополнительной информации.
        """

        self.row, self.col = (action - 1) // self.size, (action - 1) % self.size

        cords_of_free_squares = [(i, j) for i in range(self.size) for j in range(self.size)
                                  if self.board[i, j] == EMPTY]

        reward = 0

        if len(cords_of_free_squares) == 0:  # Проверяем, остались ли ещё свободные клетки для хода
            terminated = True
            if self.check_winner(player=self.current_player):
                reward = 100
        else:
            if (self.row, self.col) in cords_of_free_squares:  # Проверяем, пуста ли клетка
                self.board[self.row, self.col] = self.current_player
                if self.check_winner(player=self.current_player):
                    terminated = True
                    reward = 100
                elif len(cords_of_free_squares) == 0:  # Проверяем на ничью
                    terminated = True
                else:
                    terminated = False
                    self.current_player = O if self.current_player == X else X  # Передаём ход другому игроку
            else:
                terminated = False
                self.current_player = O if self.current_player == X else X
                reward = -100

        truncated = False
        observation = self._get_obs()
        info = self._get_info()

        self.current_turn+=1

        return observation, reward, terminated, truncated, info

def main():
    env = BoardEnv()

    episode_over = False
    while not episode_over:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        env.print_diagnostic(_get_info_dict=info, terminated=terminated)

        episode_over = terminated

if __name__ == "__main__":
    main()
