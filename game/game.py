from game_manager import GameManager
from test_classification import test

if __name__ == '__main__':
    # Testing of model before game for detection accuracy.
    test()
    # Game.
    game = GameManager()
    game.run()