game_list = ['alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis', 'bank_heist', 'battle_zone', 'beam_rider', 'bowling', 'boxing', 'breakout', 'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar', 'hero', 'ice_hockey', 'james_bond', 'kangaroo', 'krull', 'kung_fu_master', 'montezuma_revenge', 'ms_pacman', 'name_this_game', 'pong', 'private_eye', 'qbert', 'river_raid', 'road_runner', 'robotank', 'seaquest', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down', 'venture', 'video_pinball', 'wizard_of_wor', 'zaxxon']

random_score_list = [227.80, 5.80, 222.40, 210.00, 719.10, 12850.00, 14.20, 2360.00, 363.90, 23.10, 0.10, 1.70, 2090.90, 811.00, 10780.50, 152.10, -18.60, 0.00, -91.70, 0.00, 65.20, 257.60,  173.00, 1027.00, -11.20, 29.00, 52.00, 1598.00, 258.50, 0.00, 307.30, 2292.30, -20.70, 24.90, 163.90, 1338.50, 11.50, 2.20, 68.40, 148.00, 664.00, -23.80, 3568.00, 11.40, 533.40, 0.00, 16256.90, 563.50, 32.50]

human_score_list = [6875.40, 1675.80, 1496.40, 8503.30, 13156.70, 29028.10, 734.40, 37800.00, 5774.70, 154.80, 4.30, 31.80, 11963.20, 9881.80, 35410.50, 3401.30, -15.50, 309.60, 5.50, 29.60, 4334.70, 2321.00, 2672.00, 25762.50, 0.90, 406.70, 3035.00, 2394.60, 22736.20, 4366.70, 15693.40, 4076.20, 9.30, 69571.30, 13455.00, 13513.30, 7845.00, 11.90, 20181.80, 1652.30, 10250.00, -8.90, 5925.00, 167.60, 9082.00, 1187.50, 17297.60, 4756.50, 9173.30]

def get_normalized_score(env_name, returns):
    random_score = random_score_list[game_list.index(env_name)]
    human_score = human_score_list[game_list.index(env_name)]
    return (returns - random_score) / (human_score - random_score)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, required=True)
    args = parser.parse_args()

    random_score = random_score_list[game_list.index(args.env_name)]
    human_score = human_score_list[game_list.index(args.env_name)]
    print('random score: {}'.format(random_score))
    print('human score: {}'.format(human_score))
    