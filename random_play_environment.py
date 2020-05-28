import bitwise_gameboard
import time
import math
from multiprocessing.pool import ThreadPool
import csv

game = bitwise_gameboard.BitwiseGameboard()

lost = False
ai = None

number_of_random_games = 200
length_of_each_random_game = None

# Multi-threading pool and function
pool = ThreadPool(4)


# TODO: If the random games find a direct path to a win condition, simply have it choose those moves, instead of recalculating every game.
def play_random_games_on_clone(direction, game_clone):
    opening_time = time.time()
    if game_clone.simulate_move_successful(direction):
        game_clone.move(direction, print_board=False)
        result = game_clone.play_multiple_random_games(number=number_of_random_games, ai=ai)
    else:
        result = 0
    closing_time = time.time()
    print('Clone direction and score {:>6}:\t{:>9.3f},\t taking {:>6.6f}s'.format(direction, result,
                                                                                  closing_time - opening_time))
    return result


for game_number in range(0, 10000):
    iteration = 0
    start_time = time.time()
    lost = False
    game = bitwise_gameboard.BitwiseGameboard()
    with open('./games/list_of_board_scores{}.csv'.format(game_number), 'a', newline='') as scorescsvfile:
        writer = csv.writer(scorescsvfile)
        while not lost:
            print('=====ITERATION {} START====='.format(iteration))
            print('Number of random games for each clone: {}'.format(number_of_random_games))
            print('Length of each random game: {}'.format(length_of_each_random_game))
            start_time_of_this_iteration = time.time()

            # We are making four clones, each corresponding to the direction of the first move in our tree search.
            clones = {'up': game.copy(),
                      'down': game.copy(),
                      'left': game.copy(),
                      'right': game.copy()}

            # We split up the clones onto different threads to speed up computation.
            scores = {direction: pool.apply(play_random_games_on_clone, args=(direction, clone)) for (direction, clone)
                      in clones.items()}
            scores_list = [scores['up'], scores['left'], scores['down'], scores['right']]

            chosen_direction = max(scores, key=lambda key: scores[key])  # choose direction with the highest score
            print('Chosen direction: {}\tIteration: {}'.format(chosen_direction, iteration))
            board_before_move = hex(game.board)
            game.move(chosen_direction, show_score=True, print_board=True)
            lost = game.check_if_game_over()
            board_after_move = hex(game.board)

            current_time = time.time()
            time_to_choose_move = current_time - start_time_of_this_iteration

            writer.writerow([board_before_move] + scores_list + [board_after_move])

            print('Time to choose move: {}s'.format(time_to_choose_move))
            print('Time spent thus far: {}s'.format(current_time - start_time))
            print('=====ITERATION {} END====='.format(iteration))
            iteration += 1
            if iteration % 1 == 0:  # Update the number of random games to check in our search
                number_of_random_games = math.ceil(
                    -1 * 2 ** 14 / (iteration + 2 ** 14 / 120) + 120)  # This formula was found experimentally

        end_time = time.time()
        total_time_to_run_game = end_time - start_time
        print('Time to play game: {}s'.format(total_time_to_run_game))
