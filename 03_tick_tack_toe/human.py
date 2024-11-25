class HumanPlayer:

    def __init__(
            self,
            size: int
    ):

        self.size = size

    def get_action(self):
        correct = False
        while not correct:
            row_col_str = input("Введите координаты хода (строка колонка): ")
            row, col = map(int, row_col_str.split())

            if (row > self.size - 1) and (col > self.size - 1):
                print('Введите корректные координаты!')
            else:
                correct = True

        action = row * self.size + col
        return action
