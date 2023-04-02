import random
import uuid
import sqlite3
import time
import argparse

args = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--game_count', help='How many games to run.', type=int, default=10, required=False)
    parser.add_argument('-p', '--print_board', help='Print Final Board', action='store_true', default=False)
    parser.add_argument('-s', '--silent', help='Print Only Final Summary', action='store_true', default=False)
    return parser.parse_args()


class Checker:
    def __init__(self, row, col, player):
        self.row = row
        self.col = col
        self.player = player
        self.king = False

    def __repr__(self):
        return f"{self.player.upper()}{'K' if self.king else ''}"


class Board:
    def __init__(self):
        self.move_count = 0
        self.board = [[None for _ in range(8)] for _ in range(8)]
        for row in range(3):
            for col in range(8):
                if (row + col) % 2 == 0:
                    self.board[row][col] = Checker(row, col, 'black')
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 == 0:
                    self.board[row][col] = Checker(row, col, 'red')

    def print_board(self):
        extra_row_spacing = ' ' * 3
        variable_msg = f'Calculate Move: {self.move_count}'
        filler = int((27 - len(variable_msg)) / 2)
        print(f"{'*' * 27}\n{' ' * filler}{variable_msg}{' ' * filler}\n{'*' * 27}")

        column_headers = '0  1  2  3  4  5  6  7'
        print(f'{extra_row_spacing}{column_headers}')
        for row in range(8):
            print(f"{row}|", end="")
            for col in range(8):
                checker = self.board[row][col]
                if checker:
                    print(f" {checker.player[0]} ", end="")
                else:
                    print(' - ', end="")
            print("|")
        print('')

    @staticmethod
    def is_whole_number(_value):
        try:
            float_value = float(_value)
            return float_value.is_integer()
        except ValueError:
            return False

    def get_checker(self, row, col):
        return self.board[row][col]

    def set_checker(self, row, col, checker):
        self.board[row][col] = checker

    def remove_checker(self, row, col):
        self.board[row][col] = None

    def move_checker(self, row1, col1, row2, col2):
        checker = self.board[row1][col1]
        self.set_checker(row2, col2, checker)
        self.remove_checker(row1, col1)
        checker.row = row2
        checker.col = col2
        if checker.player == 'red' and row2 == 0:
            checker.king = True
        elif checker.player == 'black' and row2 == 7:
            checker.king = True

    def get_legal_moves(self, _input, _player=None, _king=None):
        """
        :param _input: `Checker` object, or Tuple of length two, containing (x, y) position
        :param _player: Boolean, must be provided if `_input` is of `tuple` type.
        :param _king: Boolean, must be provided if `_input` is of `tuple` type.
        :return: List of calculated legal Checkers moves for a given checker or position
        """
        if isinstance(_input, Checker):
            row, col = _input.row, _input.col
            player = _input.player
            if _input.king:
                directions = [(1, -1), (1, 1), (-1, -1), (-1, 1)]
            elif player == 'red':
                directions = [(-1, -1), (-1, 1)]
            else:
                directions = [(1, -1), (1, 1)]

        elif isinstance(_input, tuple):
            if not _player:
                raise f'[!] `_player` must be provided to `get_legal_moves()` function when positions are used.'
            if len(_input) % 2 != 0:
                return []
            row, col = _input[0], _input[1]
            player = _player
            if _king:
                directions = [(1, -1), (1, 1), (-1, -1), (-1, 1)]
            elif player == 'red':
                directions = [(-1, -1), (-1, 1)]
            else:
                directions = [(1, -1), (1, 1)]

        legal_moves = []
        for d_row, d_col in directions:
            row2, col2 = row + d_row, col + d_col
            if not self.is_valid_pos(row2, col2):
                continue
            if self.get_checker(row2, col2) is None:
                legal_moves.append((row2, col2))
        return legal_moves

    @staticmethod
    def is_valid_pos(row, col):
        return 0 <= row < 8 and 0 <= col < 8

    def get_legal_jumps(self, _input, _player=None, _king=None):
        """
        :param _input: `Checker` object, or Tuple of length two, containing (x, y) position
        :param _player: Boolean, must be provided if `_input` is of `tuple` type.
        :param _king: Boolean, must be provided if `_input` is of `tuple` type.
        :return: List of calculated legal Checkers jumps for a given checker or position
        """
        if isinstance(_input, Checker):
            row, col = _input.row, _input.col
            player = _input.player
            if _input.king:
                directions = [(1, -1), (1, 1), (-1, -1), (-1, 1)]
            elif player == 'red':
                directions = [(-1, -1), (-1, 1)]
            else:
                directions = [(1, -1), (1, 1)]

        elif isinstance(_input, tuple):
            if not _player:
                raise f'[!] `_player` must be provided to `get_legal_moves()` function when positions are used.'
            if len(_input) % 2 != 0:
                return []
            row, col = _input[0], _input[1]
            player = _player
            if _king:
                directions = [(1, -1), (1, 1), (-1, -1), (-1, 1)]
            elif player == 'red':
                directions = [(-1, -1), (-1, 1)]
            else:
                directions = [(1, -1), (1, 1)]

        jumps = []
        for d_row, d_col in directions:
            row2, col2 = row + d_row, col + d_col
            if not self.is_valid_pos(row2, col2):
                continue
            checker2 = self.get_checker(row2, col2)
            if checker2 is not None and checker2.player != player:
                row3, col3 = row2 + d_row, col2 + d_col
                if self.is_valid_pos(row3, col3) and self.get_checker(row3, col3) is None:
                    jumps.append((row3, col3))
        return jumps


class CheckersGame:
    def __init__(self, game_number):
        self.board = Board()
        self.current_player = 'red'
        self.current_moves = []
        self.current_jumps = []
        self.current_double_jumps = []
        self.start_time = time.time()
        self.db = sqlite3.connect('checkers.db')
        self.db.execute(
            'CREATE TABLE IF NOT EXISTS moves '
            '(move_id TEXT PRIMARY KEY, game_id TEXT, player TEXT, from_row INTEGER, from_col INTEGER, '
            'to_row INTEGER, to_col INTEGER, time_executed TEXT, red_move_count INTEGER, black_move_count INTEGER, '
            'red_score REAL, black_score REAL, red_score_change REAL, black_score_change REAL, win_magnitude REAL, '
            'current_jump_length INTEGER)')
        self.red_move_count = 0
        self.current_red_score = 0
        self.black_move_count = 0
        self.current_black_score = 0
        self.game_over = False
        self.game_id = uuid.uuid4()
        self.jump_just_taken = False
        self.jump_taken_player = None
        self.current_game_number = game_number
        self.move_count = 0
        self.winner = None
        self.game_win_time = None
        self.winner_checker_count = 0
        self.loser_checker_count = 0
        self.winning_score = 0
        self.losing_score = 0
        self.scale_factor = 1000
        self.multi_jumps = []
        self.current_checker = None
        self.current_jump_length = 0
        self.master_depth = 10
        self.red_jump_count = 0
        self.red_multi_jump_count = 0
        self.black_jump_count = 0
        self.black_multi_jump_count = 0

    def current_score(self, player_string):
        red_count, black_count = 0, 0
        for row in range(8):
            for col in range(8):
                checker = self.board.get_checker(row, col)
                if checker:
                    if checker.player == 'red':
                        red_count += 1
                    elif checker.player == 'black':
                        black_count += 1

        current_duration = time.time() - self.start_time
        red_divisor = (self.red_move_count if self.red_move_count != 0 else 1)
        black_divisor = (self.black_move_count if self.black_move_count != 0 else 1)
        red_base_score = red_count * red_divisor / current_duration / self.scale_factor
        black_base_score = black_count * black_divisor / current_duration / self.scale_factor

        if self.current_red_score == 0 and self.current_black_score == 0 and not self.jump_just_taken:
            self.current_red_score = red_base_score
            self.current_black_score = black_base_score
        else:
            if self.jump_taken_player:
                if self.jump_taken_player == 'red':
                    self.current_red_score = self.current_red_score + red_base_score
                    self.current_black_score = self.current_black_score - black_base_score
                elif self.jump_taken_player == 'black':
                    self.current_red_score = self.current_red_score - red_base_score
                    self.current_black_score = self.current_black_score + black_base_score

                self.jump_just_taken = False
                self.jump_taken_player = None

    def get_checker_counts(self):
        red_count, black_count = 0, 0
        for row in self.board.board:
            for checker in row:
                if not checker:
                    continue
                if checker.player == 'red':
                    red_count += 1
                elif checker.player == 'black':
                    black_count += 1
        return {'red': red_count, 'black': black_count}

    def play(self, _print_all=True, print_final_board=True):
        global args
        while not self.is_game_over():
            if _print_all:
                self.board.print_board()
            if self.current_player == 'red':
                self.play_player('red', print_move=_print_all)
            else:
                self.play_player('black', print_move=_print_all)
            self.current_player = 'red' if self.current_player == 'black' else 'black'
        self.game_over = True
        if print_final_board:
            self.board.print_board()
        self.winner = self.get_winner()
        if not args.silent:
            print(f"Game ID:\t{self.game_id}\nGame Result:\t{self.winner.upper()} wins!")
        remaining_checkers = self.get_checker_counts()
        self.winner_checker_count = remaining_checkers['red'] if self.winner == 'red' else remaining_checkers['black']
        self.loser_checker_count = remaining_checkers['black'] if self.winner == 'red' else remaining_checkers['red']
        self.losing_score = self.current_black_score if self.winner == 'red' else self.current_red_score
        self.winning_score = self.current_red_score if self.winner == 'red' else self.current_black_score \
                                                                            + self.losing_score

        winning_score = round(self.winning_score, 2)
        losing_score = round(self.losing_score, 2)

        if winning_score < losing_score:
            winning_score = round(abs(winning_score) + abs(losing_score), 2)
        elif winning_score == losing_score:
            winning_score += 1

        score = {
            'winner_checkers': self.winner_checker_count,
            'loser_checkers': self.loser_checker_count,
            'time_ms': round((time.time() - self.start_time) * 1000, 2),
            'winning_score': round(self.winning_score, 2),
            'losing_score': round(self.losing_score, 2),
            'win_magnitude': round(winning_score - losing_score, 2),
            'total_moves': self.move_count
        }

        player_black = {
            'player': 'BLACK',
            'jumps': self.black_jump_count,
            'multi_jumps': self.black_multi_jump_count
        }

        player_red = {
            'player': 'RED',
            'jumps': self.red_jump_count,
            'multi_jumps': self.red_multi_jump_count
        }

        if not args.silent:
            print(f'{"-" * len(str(score))}\n{score}\n{"-" * len(str(score))}')
            print(f'{player_red}')
            print(f'{player_black}\n\n{"+" * len(str(player_black))}\n')

    def estimate_decision_time(self):
        decision_time = 0.0001  # Define standard unit of time per decision here.
        duration = time.time() - self.start_time
        if duration > (decision_time * 60):  # Increase decision time based on the current game time
            decision_time += (decision_time * 0.5)
        elif duration > (decision_time * 120):
            decision_time += (decision_time * 1.0)
        # Increase decision time based on the number of available jumps and double jumps
        decision_time += (decision_time * 0.1) * len(self.current_jumps)
        decision_time += (decision_time * 0.1) * len(self.multi_jumps)
        decision_time *= random.uniform(0.9, 1.1)  # Add randomness to the decision time to simulate human variability
        return decision_time

    def play_player(self, _player,  print_move=True):
        self.current_moves = self.get_all_moves(_player.lower())
        self.current_jumps = self.get_all_jumps(_player.lower())
        if self.current_jumps:
            self.multi_jumps = []
            for current_jump in self.current_jumps:
                self.current_checker = self.board.get_checker(current_jump[0], current_jump[1])
                self.get_legal_multi_jumps(current_jump, _player, self.current_checker.king,
                                           max_depth=self.master_depth)
                if self.multi_jumps:
                    move = self.choose_multi_jump(self.multi_jumps, _player)
                    if _player.upper() == 'RED':
                        self.red_multi_jump_count += 1
                    elif _player.upper() == 'BLACK':
                        self.black_multi_jump_count += 1
                else:
                    move = self.choose_jump(self.current_jumps, _player)
                    if _player.upper() == 'RED':
                        self.red_jump_count += 1
                    elif _player.upper() == 'BLACK':
                        self.black_jump_count += 1
                self.current_checker = None
        elif self.current_moves:
            move = self.choose_move(self.current_moves)
        else:
            if print_move and not args.silent:
                print("No legal moves.")
            move = None
        if move:
            self.make_move(_player.lower(), move, _print=print_move)
            time.sleep(self.estimate_decision_time())
            if _player == 'red':
                self.red_move_count += 1
            elif _player == 'black':
                self.black_move_count += 1

    def get_legal_multi_jumps(self, _jump_tuple, _player, _king, max_depth=None, current_depth=0):
        if max_depth is not None and current_depth >= self.master_depth + 1:
            self.current_jump_length = 0
            return

        start_position = _jump_tuple[-2], _jump_tuple[-1]
        legal_multi_jumps_list = self.board.get_legal_jumps(start_position, _player=_player, _king=_king)

        if legal_multi_jumps_list:
            for legal_multi_jump in legal_multi_jumps_list:
                extended_tuple = _jump_tuple + (legal_multi_jump[-2], legal_multi_jump[-1])
                self.multi_jumps.append(extended_tuple)
                self.get_legal_multi_jumps(extended_tuple, _player=_player, _king=_king, max_depth=max_depth,
                                           current_depth=current_depth + 1)
        else:
            return

    def get_all_moves(self, player):
        moves = []
        for row in range(8):
            for col in range(8):
                checker = self.board.get_checker(row, col)
                if checker:
                    if checker.player == player:
                        moves.extend([(row, col, row2, col2) for row2, col2 in self.board.get_legal_moves(checker)])
        return moves

    def get_all_jumps(self, player):
        jumps = []
        for row in range(8):
            for col in range(8):
                checker = self.board.get_checker(row, col)
                if checker:
                    if checker.player == player:
                        jumps.extend([(row, col, row2, col2) for row2, col2 in self.board.get_legal_jumps(checker)])
        return jumps

    @staticmethod
    def choose_move(moves):
        return random.choice(moves)

    def choose_jump(self, jumps, _player_string):
        self.jump_just_taken = True
        self.jump_taken_player = _player_string
        return random.choice(jumps)

    def choose_multi_jump(self, multi_jumps, _player_string):
        self.jump_just_taken = True
        self.jump_just_taken = _player_string
        longest_tuple = self.longest_tuple(multi_jumps)
        self.current_jump_length = (len(longest_tuple) - 2) / 2
        return self.longest_tuple(multi_jumps)

    @staticmethod
    def longest_tuple(tuples_list):
        if not tuples_list:
            return None
        max_length = max(len(t) for t in tuples_list)
        longest_tuples = [t for t in tuples_list if len(t) == max_length]
        return random.choice(longest_tuples)

    def make_move(self, player, move, _print=True):
        if (len(move) - 2) % 2 != 0:
            print(f'[!] Failed move because tuple length is odd: `{player}` --> `{move}`')
            return
        from_row, from_col, to_row, to_col = move[0], move[1], move[2], move[3]
        self.board.move_checker(from_row, from_col, to_row, to_col)
        if abs(from_row - to_row) == 2:
            jumped_row, jumped_col = (from_row + to_row) // 2, (from_col + to_col) // 2
            self.board.remove_checker(jumped_row, jumped_col)
        self.record_move(player, from_row, from_col, to_row, to_col, _print=_print)
        self.board.move_count += 1
        if len(move) > 4:
            self.make_move(player, tuple(list(move)[2:]))

    def record_move(self, player, from_row, from_col, to_row, to_col, _print=True):
        move_id = f'{self.game_id}-{self.move_count}'
        red_score_change, black_score_change = self.current_red_score, self.current_black_score
        self.current_score('red')
        red_score_change = round(self.current_red_score - red_score_change, 4)
        black_score_change = round(self.current_black_score - black_score_change, 4)
        self.game_win_time = f'{time.time()}'

        if self.current_jump_length >= self.master_depth:
            self.current_jump_length = 0

        if self.current_jump_length != 0:
            player_test = player.upper()
            if player_test == 'RED':
                modifier = red_score_change * self.current_jump_length
                red_score_change = red_score_change + modifier
                self.current_red_score += modifier
            elif player_test == 'BLACK':
                modifier = red_score_change * self.current_jump_length
                black_score_change = red_score_change + modifier
                self.current_red_score += modifier

        if _print:
            print(f'[-] Game Number: {self.current_game_number}')
            print(f'[-] {player.upper()} chooses to move: ({from_row}, {from_col}) --> ({to_row}, {to_col})\n')

        self.db.execute('INSERT INTO moves (move_id, game_id, player, from_row, from_col, to_row, to_col, '
                        'time_executed, red_move_count, black_move_count, red_score, black_score, '
                        'red_score_change, black_score_change, win_magnitude, current_jump_length) '
                        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                        (move_id, f'{self.game_id}', player, from_row, from_col, to_row, to_col,
                         self.game_win_time, self.red_move_count, self.black_move_count,self.current_red_score,
                         self.current_black_score, red_score_change, black_score_change,
                         abs(self.current_black_score - self.current_red_score), self.current_jump_length))
        self.db.commit()
        self.move_count += 1

    def is_game_over(self):
        return not self.get_all_moves('red') or not self.get_all_moves('black')

    def get_winner(self):
        num_red_checkers = 0
        num_black_checkers = 0
        for row in range(8):
            for col in range(8):
                checker = self.board.get_checker(row, col)
                if checker:
                    if checker.player == 'red':
                        num_red_checkers += 1
                    else:
                        num_black_checkers += 1
        if num_red_checkers > num_black_checkers:
            self.winner = 'red'
            return 'red'
        elif num_black_checkers > num_red_checkers:
            self.winner = 'black'
            return 'black'
        else:
            self.winner = None
            return 'tie'


class Backend:
    def __init__(self):
        global args
        self.start_time = time.time()
        self.args = parse_args()
        args = self.args
        self.db = sqlite3.connect('checkers.db')
        self.db.execute(
            'CREATE TABLE IF NOT EXISTS games '
            '(game_id TEXT PRIMARY KEY, winning_player TEXT, time_game_completed TEXT, red_move_count INTEGER, '
            'black_move_count INTEGER, red_score REAL, black_score REAL, '
            'winner_checker_count INTEGER, loser_checker_count INTEGER)')
        self.game_count = self.args.game_count
        self.current_game_number = 0
        self.execute_loop(self.game_count)
        self.current_game = None
        self.final_msg = self.generate_final_msg()

    def generate_final_msg(self, print_result=True):
        delimiter = '*'
        current_duration = time.time() - self.start_time
        if 60 <= current_duration < 3600:
            current_duration = f'{round(current_duration / 60, 2)} minutes'
        elif 3600 <= current_duration < 86400:
            current_duration = f'{round(current_duration / 3600, 2)} hours'
        elif current_duration >= 86400:
            current_duration = f'{round(current_duration / 86400, 2)} days'
        else:
            current_duration = f'{current_duration} seconds'
        msg = f'Completed `{self.game_count}` games in `{current_duration}`.\n'
        msg_header = ' FINAL RESULTS '
        msg_header_fill_len = int(round((len(msg) - len(msg_header)) / 2, 0))
        msg_full_header = f'{delimiter * msg_header_fill_len}{msg_header}{delimiter * msg_header_fill_len}'
        final_msg = f'{msg_full_header}\n{msg}{delimiter * len(msg_full_header)}'
        if print_result:
            print(final_msg)
        return final_msg

    def execute_loop(self, loop_count):
        for game_number in range(1, loop_count + 1):
            self.current_game_number += 1
            self.current_game = CheckersGame(self.current_game_number)
            print(f'[+] Game Number: {game_number}/{loop_count}\n')
            self.current_game.play(print_final_board=self.args.print_board)
            self.record_game_end()

    def record_game_end(self):
        self.db.execute('INSERT INTO games (game_id, winning_player, time_game_completed, red_move_count, '
                        'black_move_count, red_score, black_score, winner_checker_count, loser_checker_count) '
                        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                        (f'{self.current_game.game_id}', self.current_game.winner, self.current_game.game_win_time,
                         self.current_game.red_move_count, self.current_game.black_move_count,
                         self.current_game.current_red_score, self.current_game.current_black_score,
                         self.current_game.winner_checker_count, self.current_game.loser_checker_count))
        self.db.commit()


if __name__ == '__main__':
    backend = Backend()
