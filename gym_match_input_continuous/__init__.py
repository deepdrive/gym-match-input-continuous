from gym.envs.registration import register

register(
    id='match-input-continuous-v0',
    entry_point='gym_match_input_continuous.envs:MatchInputContinuousEnv',
)

register(
    id='match-pairs-continuous-v0',
    entry_point='gym_match_input_continuous.envs:MatchPairsContinuousEnv',
)

register(
    id='corrective-psuedo-steering-v0',
    entry_point='gym_match_input_continuous.envs:CorrectivePsuedoSteeringEnv',
)
