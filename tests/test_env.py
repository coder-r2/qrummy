"""RLCard env (SPEC §5.1-§5.3, §10.2)."""

import numpy as np
import pytest
import rlcard

import indian_rummy.rlcard_env  # noqa: F401
from indian_rummy import actions as A
from indian_rummy.agents import HeuristicAgent, RandomAgent
from indian_rummy.features import OBS_DIM, analyse, encode
from indian_rummy.game import RummyGame


def test_env_shapes():
    env = rlcard.make("indian-rummy", config={"seed": 0})
    assert env.num_actions == A.NUM_ACTIONS == 56
    assert env.state_shape == [[OBS_DIM], [OBS_DIM]]
    state, pid = env.reset()
    assert state["obs"].shape == (OBS_DIM,) and state["obs"].dtype == np.float32
    assert list(state["legal_actions"]) == env.game.get_legal_actions()


@pytest.mark.slow
def test_env_runs_random_agents_1000_rounds():
    env = rlcard.make("indian-rummy", config={"seed": 1, "game_turn_cap": 120})
    env.set_agents([RandomAgent(seed=1, allow_forfeit=False), RandomAgent(seed=2, allow_forfeit=False)])
    for _ in range(1000):
        traj, payoffs = env.run(is_training=False)
        assert env.game.is_over()
        assert -1.0 <= payoffs[0] <= 1.0 and abs(payoffs[0] + payoffs[1]) < 1e-6
        assert len(traj[0]) % 2 == 1 and len(traj[1]) % 2 == 1


def test_heuristic_never_illegal():
    env = rlcard.make("indian-rummy", config={"seed": 2})
    h = HeuristicAgent()
    for _ in range(100):
        state, pid = env.reset()
        while not env.is_over():
            a = h.step(state)
            assert a in state["legal_actions"]
            state, pid = env.step(a)


def test_obs_does_not_depend_on_hidden_opponent_cards():
    """Swapping the opponent's hidden (non-public) cards with draw-pile cards leaves our obs unchanged."""
    g = RummyGame(seed=5, reveal_policy={0: lambda g, p, o: o[0], 1: lambda g, p, o: o[0]})
    g.init_game()
    me = g.current
    st = g.get_state(me)
    obs1 = encode(analyse(st, g.types, g.variant))
    opp = g.players[1 - me]
    hidden = [c for c in opp.hand if c not in opp.locked and c not in opp.known_open]
    swap_n = min(5, len(hidden))
    for i in range(swap_n):
        j = len(g.draw_pile) - 1 - i
        k = opp.hand.index(hidden[i])
        opp.hand[k], g.draw_pile[j] = g.draw_pile[j], opp.hand[k]
    obs2 = encode(analyse(g.get_state(me), g.types, g.variant))
    assert np.array_equal(obs1, obs2)
