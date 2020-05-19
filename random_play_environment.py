import bitwise_gameboard
import time

game = bitwise_gameboard.BitwiseGameboard()

lost = False
ai = None

number_of_random_games = 100
length_of_each_random_game = None

iteration = 0
start_time = time.time()
while not lost:
    print('=====ITERATION {} START====='.format(iteration))
    start_time_of_this_iteration = time.time()

    clones = {'up': game.copy(),
              'down': game.copy(),
              'left': game.copy(),
              'right': game.copy()}
    scores = dict()
    for direction, clone in clones.items():
        clone.move(direction, print_board=False)
        scores[direction] = clone.play_multiple_random_games(number=number_of_random_games, ai=ai)
        # print('Clone that moved in direction: {}'.format(direction))
        # clone.print(show_score=True)
        print('Clone that moved {} got an average score of {}'.format(direction, scores[direction]))

    chosen_direction = max(scores, key=lambda key: scores[key])  # choose direction with the highest score
    print('Chosen direction: {}\tIteration: {}'.format(chosen_direction, iteration))
    game.move(chosen_direction)
    lost = game.is_board_full()

    current_time = time.time()
    time_to_choose_move = current_time - start_time_of_this_iteration
    print('Time to choose move: {}s'.format(time_to_choose_move))
    print('=====ITERATION {} END====='.format(iteration))
    iteration += 1

end_time = time.time()
total_time_to_run_game = end_time - start_time
print('Time to play game: {}s'.format(total_time_to_run_game))
