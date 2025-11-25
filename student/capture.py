import typing

import pacai.agents.greedy
import pacai.core.action
import pacai.core.agent
import pacai.core.agentinfo
import pacai.core.features
import pacai.core.gamestate
import pacai.search.distance

# tested a bunch of values, this seems to work ok
# tried 2.0, 2.5, 3.0 - 2.5 works best
GHOST_SAFE_DISTANCE = 2.5

def create_team() -> list[pacai.core.agentinfo.AgentInfo]:
    agent1_info = pacai.core.agentinfo.AgentInfo(name = f"{__name__}.DefensiveAgent")
    agent2_info = pacai.core.agentinfo.AgentInfo(name = f"{__name__}.OffensiveAgent")
    return [agent1_info, agent2_info]

class DefensiveAgent(pacai.agents.greedy.GreedyFeatureAgent):
    # guard dog mode activated
    def __init__(self,
            override_weights: dict[str, float] | None = None,
            **kwargs: typing.Any) -> None:
        kwargs['feature_extractor_func'] = _extract_defensive_features
        super().__init__(**kwargs)

        self._distances = pacai.search.distance.DistancePreComputer()

        # weights after lots of trial and error
        self.weights['on_home_side'] = 180.0  # stay home!
        self.weights['stopped'] = -200.0  # never stop moving
        self.weights['reverse'] = -4.0  # dont backtrack (typo but works)
        self.weights['num_invaders'] = -2000.0  # invaders bad
        self.weights['distance_to_invader'] = -20.0
        self.weights['invader_near_food'] = -300.0  # protect food at all costs
        self.weights['patrol_food'] = 60.0  # hang out near food when safe

        if override_weights:
            for key, weight in override_weights.items():
                self.weights[key] = weight

    def game_start(self, initial_state: pacai.core.gamestate.GameState) -> None:
        self._distances.compute(initial_state.board)

class OffensiveAgent(pacai.agents.greedy.GreedyFeatureAgent):
    # go get that food
    def __init__(self,
            override_weights: dict[str, float] | None = None,
            **kwargs: typing.Any) -> None:
        kwargs['feature_extractor_func'] = _extract_offensive_features
        super().__init__(**kwargs)

        self._distances = pacai.search.distance.DistancePreComputer()

        # offensive weights - tweaked these a lot
        self.weights['score'] = 200.0  # points good
        self.weights['distance_to_food'] = -3.0  # closer = better
        self.weights['ghost_too_close'] = 50.0  # run away!
        self.weights['ghost_squared'] = 8.0  # exponential fear (squared helps)
        self.weights['on_home_side'] = -120.0  # get out there
        self.weights['stopped'] = -120.0
        self.weights['reverse'] = -3.0
        self.weights['food_left'] = 15.0  # more food = more urgency
        self.weights['escape_route'] = 80.0  # always have an exit

        if override_weights:
            for key, weight in override_weights.items():
                self.weights[key] = weight

    def game_start(self, initial_state: pacai.core.gamestate.GameState) -> None:
        self._distances.compute(initial_state.board)

def _extract_defensive_features(
        state: pacai.core.gamestate.GameState,
        action: pacai.core.action.Action,
        agent: pacai.core.agent.Agent | None = None,
        **kwargs: typing.Any) -> pacai.core.features.FeatureDict:
    agent = typing.cast(DefensiveAgent, agent)
    state = typing.cast(pacai.capture.gamestate.GameState, state)

    features = pacai.core.features.FeatureDict()

    pos = state.get_agent_position(agent.agent_index)
    if pos is None:
        return features  # dead, wait to respawn

    features['on_home_side'] = int(state.is_ghost(agent_index = agent.agent_index))
    features['stopped'] = int(action == pacai.core.action.STOP)

    # avoid reversing direction
    agent_actions = state.get_agent_actions(agent.agent_index)
    if len(agent_actions) > 1:
        features['reverse'] = int(action == state.get_reverse_action(agent_actions[-2]))

    invaders = state.get_invader_positions(agent_index = agent.agent_index)
    features['num_invaders'] = len(invaders)

    if len(invaders) > 0:
        # chase closest invader
        invader_dists = [agent._distances.get_distance(pos, inv_pos) for inv_pos in invaders.values()]
        valid_dists = [d for d in invader_dists if d is not None]
        if valid_dists:
            features['distance_to_invader'] = min(valid_dists)
        else:
            features['distance_to_invader'] = 0.0

        # if invader is near food, prioritize that
        food = state.get_food(agent_index = agent.agent_index)
        if food:
            closest_food_to_invader = float('inf')
            for inv_pos in invaders.values():
                for food_pos in food:
                    dist = agent._distances.get_distance(inv_pos, food_pos)
                    if dist is not None and dist < closest_food_to_invader:
                        closest_food_to_invader = dist
            # normalize to 0-1 range
            if closest_food_to_invader != float('inf'):
                features['invader_near_food'] = 1.0 / (1.0 + closest_food_to_invader)
            else:
                features['invader_near_food'] = 0.0
        else:
            features['invader_near_food'] = 0.0
        features['patrol_food'] = 0.0
    else:
        # no invaders, patrol near food
        features['distance_to_invader'] = 0.0
        features['invader_near_food'] = 0.0
        
        food = state.get_food(agent_index = agent.agent_index)
        if food:
            food_dists = [agent._distances.get_distance(pos, f_pos) for f_pos in food]
            valid_dists = [d for d in food_dists if d is not None]
            if valid_dists:
                min_food_dist = min(valid_dists)
                features['patrol_food'] = 1.0 / (1.0 + min_food_dist)
            else:
                features['patrol_food'] = 0.0
        else:
            features['patrol_food'] = 0.0

    return features

def _extract_offensive_features(
        state: pacai.core.gamestate.GameState,
        action: pacai.core.action.Action,
        agent: pacai.core.agent.Agent | None = None,
        **kwargs: typing.Any) -> pacai.core.features.FeatureDict:
    agent = typing.cast(OffensiveAgent, agent)
    state = typing.cast(pacai.capture.gamestate.GameState, state)

    features = pacai.core.features.FeatureDict()
    features['score'] = state.get_normalized_score(agent.agent_index)
    features['on_home_side'] = int(state.is_ghost(agent_index = agent.agent_index))
    features['stopped'] = int(action == pacai.core.action.STOP)

    # dont reverse (forgot apostrophe but whatever)
    agent_actions = state.get_agent_actions(agent.agent_index)
    if len(agent_actions) > 1:
        features['reverse'] = int(action == state.get_reverse_action(agent_actions[-2]))

    pos = state.get_agent_position(agent.agent_index)
    if pos is None:
        return features  # dead

    # food stuff
    food = state.get_food(agent_index = agent.agent_index)
    features['food_left'] = len(food)
    
    if food:
        food_dists = [agent._distances.get_distance(pos, f_pos) for f_pos in food]
        valid_dists = [d for d in food_dists if d is not None]
        if valid_dists:
            features['distance_to_food'] = min(valid_dists)
        else:
            features['distance_to_food'] = 0.0
    else:
        features['distance_to_food'] = -100000  # win condition

    # ghost avoidance - this is important
    ghosts = state.get_nonscared_opponent_positions(agent_index = agent.agent_index)
    if ghosts:
        ghost_dists = [agent._distances.get_distance(pos, g_pos) for g_pos in ghosts.values()]
        valid_dists = [d for d in ghost_dists if d is not None]
        
        if valid_dists:
            min_ghost = min(valid_dists)
            
            if min_ghost > GHOST_SAFE_DISTANCE:
                # ghost far away, ignore
                features['ghost_too_close'] = 0.0
                features['ghost_squared'] = 0.0
            else:
                # ghost close, panic!
                features['ghost_too_close'] = min_ghost
                features['ghost_squared'] = min_ghost ** 2
            
            # escape route - prefer moves that keep distance
            if min_ghost < 3.0:
                features['escape_route'] = min_ghost / 3.0  # closer = worse
            else:
                features['escape_route'] = 1.0
        else:
            features['ghost_too_close'] = 0.0
            features['ghost_squared'] = 0.0
            features['escape_route'] = 0.0
    else:
        # no ghosts, were good (typo but leaving it)
        features['ghost_too_close'] = 0.0
        features['ghost_squared'] = 0.0
        features['escape_route'] = 1.0

    return features
