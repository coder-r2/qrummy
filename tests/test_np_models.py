"""NumPy inference matches the torch models it was exported from (D-085)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from rlcard.agents.dqn_agent import EstimatorNetwork  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import export_models  # noqa: E402

from indian_rummy.agents.np_models import NpDQNPolicy, NpValueModel  # noqa: E402
from indian_rummy.agents.value import ValueNet  # noqa: E402
from indian_rummy.features import FEATURE_VERSION, OBS_DIM  # noqa: E402


def _randomise_bn(bn):
    with torch.no_grad():
        bn.running_mean.uniform_(-0.5, 0.5)
        bn.running_var.uniform_(0.5, 2.0)
        bn.weight.uniform_(0.5, 1.5)
        bn.bias.uniform_(-0.2, 0.2)


def test_dqn_export_matches(tmp_path):
    net = EstimatorNetwork(56, [OBS_DIM], [64, 32])
    _randomise_bn(net.fc_layers[1])
    meta = {"obs_dim": OBS_DIM, "mlp_layers": [64, 32], "num_actions": 56, "feature_version": FEATURE_VERSION}
    torch.save({"qnet": net.state_dict(), "meta": meta}, tmp_path / "q.pt")
    export_models.export_dqn(str(tmp_path / "q.pt"), str(tmp_path / "q.npz"))
    pol = NpDQNPolicy.load(str(tmp_path / "q.npz"))
    x = export_models.probe_inputs(OBS_DIM)
    with torch.no_grad():
        ref = net.eval()(torch.from_numpy(x)).numpy()
    assert np.abs(pol.q_values(x) - ref).max() < 1e-4
    assert pol.masked_q(x[0], [0, 1]).keys() == {0, 1}


def test_value_export_matches(tmp_path):
    net = ValueNet(OBS_DIM, (64, 32))
    meta = {"obs_dim": OBS_DIM, "layers": [64, 32], "feature_version": FEATURE_VERSION, "temperature": 1.7}
    torch.save({"state_dict": net.state_dict(), "meta": meta}, tmp_path / "v.pth")
    export_models.export_value(str(tmp_path / "v.pth"), str(tmp_path / "v.npz"))
    vm = NpValueModel.load(str(tmp_path / "v.npz"))
    assert vm.temperature == pytest.approx(1.7)
    p, s = vm.predict(export_models.probe_inputs(OBS_DIM)[0])
    assert 0.0 <= p <= 1.0 and -1.0 <= s <= 1.0


def test_feature_version_guard(tmp_path):
    net = ValueNet(OBS_DIM, (8,))
    meta = {"obs_dim": OBS_DIM, "layers": [8], "feature_version": FEATURE_VERSION}
    torch.save({"state_dict": net.state_dict(), "meta": meta}, tmp_path / "v.pth")
    export_models.export_value(str(tmp_path / "v.pth"), str(tmp_path / "v.npz"))
    import json
    with np.load(tmp_path / "v.npz") as z:
        spec = json.loads(str(z["spec"]))
        arrays = {k: z[k] for k in z.files if k != "spec"}
    spec["meta"]["feature_version"] = FEATURE_VERSION - 1
    np.savez(tmp_path / "old.npz", spec=np.array(json.dumps(spec)), **arrays)
    with pytest.raises(ValueError):
        NpValueModel.load(str(tmp_path / "old.npz"))
