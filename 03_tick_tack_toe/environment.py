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

        self.action_space = spaces.Discrete(size * size)
        self.observation_space = spaces.MultiDiscrete([X, O, EMPTY] * (size * size))

        self.reset()

    def _get_obs(self) -> tuple:
        """Преобразуем состояние среды (environment) в наблюдение (observation).
        Эти данные подаются на обучение нейронной сети

        :return: кортеж-наблюдение
        """

        board = list(self.board.flatten())
        current_player = X if self.current_player == O else O

        return tuple(np.array(board + [current_player]))

    def _get_info(self) -> dict:
        """Диагностические данные, которые позволяют отслеживать работу нейронной сети

        :return: словарь диагностической информации
        """
        vectorized_chr = np.vectorize(chr)
        human_readable_board = vectorized_chr(self.board)

        human_readable_current_player = chr(self.current_player)

        return {
            'board': human_readable_board,
            'current_player': human_readable_current_player,
            'current_turn': self.current_turn
        }

    def print_diagnostic(self, _get_info_dict: dict, terminated: bool, truncated: bool):
        """Функция человеко-читаемого вывода поля (и консольный интерфейс для игры человека и обученного бота)

        :param _get_info_dict: словарь диагностической информации
        :param terminated: победил ли кто-нибудь
        :param truncated: ничья ли
        :return:
        """

        if self.row != -1 and self.col != -1:  # проверка, что это не начало игры

            print(f'Ход #{_get_info_dict['current_turn']}', end=', ')
            print('ходит', end=' ')
            if terminated:
                print('первый игрок,' if _get_info_dict['current_player'] == 'X' else 'второй игрок,', end=' ')
            else:
                print('первый игрок,' if _get_info_dict['current_player'] == 'O' else 'второй игрок,', end=' ')
            print(f'ход: {(int(self.row), int(self.col))}')

            print('Поле после данного хода: ')
            human_readable_board = _get_info_dict['board']

            for row in human_readable_board:
                for symbol in row:
                    print(symbol, end=' ')
                print('')

            if truncated:
                print('Ничья!')

            if terminated:
                print(f'Победа игрока {_get_info_dict['current_player']}!')

            print()
        else:
            print("Игра начинается. ")
            print('Поле пусто: ')

            human_readable_board = _get_info_dict['board']

            for row in human_readable_board:
                for symbol in row:
                    print(symbol, end=' ')
                print('')

            print()

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

    def step(self, action: int) -> Tuple[tuple, int, bool, bool, dict]:
        """Выполняет действие, вычисляет новое состояние среды и возвращает кортеж из следующего наблюдения,
        вознаграждения, признаков завершения и дополнительной информации.

        :param action: Действие, выполняемое агентом.
        :return: Кортеж из следующего наблюдения, вознаграждения, признаков завершения и дополнительной информации.
        """

        self.row, self.col = action // self.size, action % self.size

        terminated = False
        truncated = False
        reward = 0

        if self.board[self.row, self.col] == EMPTY:  # Проверяем, пуста ли клетка
            self.board[self.row, self.col] = self.current_player
            if self.check_winner(player=self.current_player):
                terminated = True
                self.current_turn += 1
                reward = 1000
            else:
                self.current_turn += 1
                self.current_player = O if self.current_player == X else X  # Передаём ход другому игроку
                reward = -1
        else:
            reward = -1_000

        check_of_free_squares = np.any(np.isin(self.board, EMPTY))

        if not check_of_free_squares:  # Проверяем, остались ли ещё свободные клетки для хода
            truncated = True

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
