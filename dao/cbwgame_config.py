"""
Configuration for the gym-idsgame environment
"""


class IdsGameConfig:
    """
    Data Object representing the configuration of the cbwgame_config:
    """
    def __init__(self):
        """
        Constructor, initializes the config
        :param render_config: render config, e.g colors, size, line width etc.
        :param game_config: game configuration, e.g. number of nodes
        :param defender_agent: the defender agent
        :param attacker_agent: the attacker agent
        :param initial_state_path: path to the initial state
        :param save_trajectories: boolean flag whether trajectories should be saved to create a self-play-dataset
        :param save_attack_stats: boolean flag whether to save attack statistics or not
        """
        pass