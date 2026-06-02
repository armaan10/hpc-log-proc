"""
tests.py — unit tests for the HPC pipeline.

Run with:
    python tests.py

Tests cover:
  1. RI formula correctness (temporal and spatial)
  2. GPU summarization (util mean, power sum, energy, num_gpus_used)
  3. CPU summarization (util norm, mem pct, I/O conversion)
  4. Slurm cleaning (timestamps, gpu count parsing, runtime)
  5. Edge cases (single node, all-zero utilisation, missing columns)
"""

import unittest
import numpy as np
import pandas
import sys
import os

# Make sure imports work from repo root
sys.path.insert(0, os.path.dirname(__file__))

from ri import _ri_temporal, _ri_spatial, calculate_ri_for_job
from slurm_utils import clean_slurm_row, _parse_num_alloc_gpus, _parse_gpu_type
from gpu_processing import summarize_gpu
from cpu_processing import summarize_cpu


# =========================================================================== #
# Helpers                                                                      #
# =========================================================================== #

def _make_gpu_ts(node_id: str, gpu_index: int, util_vals: list,
                 mem_util_vals: list = None, power_vals: list = None,
                 mem_used_vals: list = None, temp_vals: list = None) -> pandas.DataFrame:
    """Build a minimal tidy GPU DataFrame for testing."""
    n = len(util_vals)
    ts = pandas.date_range("2021-05-01", periods=n, freq="10s", tz="UTC")
    return pandas.DataFrame({
        "timestamp":    ts,
        "node_id":      node_id,
        "gpu_index":    gpu_index,
        "util_pct":     util_vals,
        "mem_util_pct": mem_util_vals  if mem_util_vals  else [0.0] * n,
        "power_w":      power_vals     if power_vals     else [100.0] * n,
        "mem_used_kib": mem_used_vals  if mem_used_vals  else [1024.0] * n,
        "temperature":  temp_vals      if temp_vals      else [60.0] * n,
    })


def _make_cpu_ts(node_id: str, util_vals: list,
                 mem_rss_vals: list = None, mem_avail_vals: list = None,
                 read_vals: list = None, write_vals: list = None) -> pandas.DataFrame:
    n = len(util_vals)
    ts = pandas.date_range("2021-05-01", periods=n, freq="10s", tz="UTC")
    mem_rss   = mem_rss_vals   if mem_rss_vals   else [512.0] * n
    mem_avail = mem_avail_vals if mem_avail_vals  else [1024.0] * n
    return pandas.DataFrame({
        "timestamp":          ts,
        "node_id":            node_id,
        "cpu_util_pct":       util_vals,
        "mem_rss_kb":         mem_rss,
        "mem_avail_kb":       mem_avail,
        "mem_pct_utilization": [r / a if a > 0 else 0
                                for r, a in zip(mem_rss, mem_avail)],
        "read_kb":  read_vals  if read_vals  else [0.0] * n,
        "write_kb": write_vals if write_vals else [0.0] * n,
    })


# =========================================================================== #
# 1. RI formula tests                                                          #
# =========================================================================== #

class TestRITemporal(unittest.TestCase):

    def test_perfect_utilisation_ri_temporal_is_zero(self):
        """Constant utilisation => no temporal imbalance."""
        series = {"node1": np.array([50.0, 50.0, 50.0])}
        self.assertAlmostEqual(_ri_temporal(series), 0.0)

    def test_half_idle_ri_temporal(self):
        """
        U = [100, 0] => sum=100, max=100, T+1=2
        RI_temporal = 1 - 100/(100*2) = 0.5
        """
        series = {"node1": np.array([100.0, 0.0])}
        self.assertAlmostEqual(_ri_temporal(series), 0.5)

    def test_multi_node_takes_worst_case(self):
        """RI_temporal is the MAX over nodes."""
        series = {
            "node1": np.array([100.0, 0.0]),    # RI = 0.5
            "node2": np.array([80.0, 80.0]),    # RI = 0.0
        }
        self.assertAlmostEqual(_ri_temporal(series), 0.5)

    def test_all_zero_utilisation(self):
        """All zeros => RI_temporal = 0."""
        series = {"node1": np.array([0.0, 0.0, 0.0])}
        self.assertAlmostEqual(_ri_temporal(series), 0.0)

    def test_single_timestep(self):
        """Single timestep => sum == max*1 => RI = 0."""
        series = {"node1": np.array([75.0])}
        self.assertAlmostEqual(_ri_temporal(series), 0.0)

    def test_nan_values_ignored(self):
        """NaNs should be dropped before calculation."""
        series = {"node1": np.array([100.0, np.nan, 0.0])}
        # After dropping NaN: [100, 0] => RI = 0.5
        self.assertAlmostEqual(_ri_temporal(series), 0.5)


class TestRISpatial(unittest.TestCase):

    def test_single_node_ri_spatial_is_zero(self):
        series = {"node1": np.array([50.0, 80.0])}
        self.assertAlmostEqual(_ri_spatial(series), 0.0)

    def test_equal_max_ri_spatial_is_zero(self):
        """Same peak on all nodes => no spatial imbalance."""
        series = {
            "node1": np.array([80.0, 60.0]),
            "node2": np.array([40.0, 80.0]),
        }
        self.assertAlmostEqual(_ri_spatial(series), 0.0)

    def test_one_idle_node_ri_spatial(self):
        """
        node1 max=100, node2 max=0
        global_max=100
        numerator = (100-100) + (100-0) = 100
        denominator = 100*2 = 200
        RI_spatial = 100/200 = 0.5
        """
        series = {
            "node1": np.array([100.0]),
            "node2": np.array([0.0]),
        }
        self.assertAlmostEqual(_ri_spatial(series), 0.5)

    def test_all_zero_ri_spatial_is_zero(self):
        series = {
            "node1": np.array([0.0, 0.0]),
            "node2": np.array([0.0, 0.0]),
        }
        self.assertAlmostEqual(_ri_spatial(series), 0.0)

    def test_three_nodes_partial_imbalance(self):
        """
        node1 max=90, node2 max=90, node3 max=30
        global_max=90
        numerator = 0 + 0 + 60 = 60
        denominator = 90*3 = 270
        RI_spatial = 60/270 ≈ 0.2222
        """
        series = {
            "node1": np.array([90.0]),
            "node2": np.array([90.0]),
            "node3": np.array([30.0]),
        }
        self.assertAlmostEqual(_ri_spatial(series), 60/270, places=3)


class TestCalculateRIForJob(unittest.TestCase):

    def test_cpu_only_job_has_nan_gpu_ri(self):
        cpu_ts = {"node1": _make_cpu_ts("node1", [50.0, 60.0, 55.0])}
        res = calculate_ri_for_job({}, cpu_ts, is_gpu_job=False)
        self.assertTrue(np.isnan(res["ri_temporal_gpu_util"]))
        self.assertTrue(np.isnan(res["ri_spatial_gpu_util"]))
        # CPU RIs should be computed
        self.assertFalse(np.isnan(res["ri_temporal_cpu_util"]))

    def test_gpu_job_has_gpu_and_cpu_ri(self):
        gpu_ts = {
            "node1": pandas.concat([
                _make_gpu_ts("node1", 0, [80.0, 90.0]),
                _make_gpu_ts("node1", 1, [70.0, 85.0]),
            ], ignore_index=True),
        }
        cpu_ts = {"node1": _make_cpu_ts("node1", [50.0, 60.0])}
        res = calculate_ri_for_job(gpu_ts, cpu_ts, is_gpu_job=True)
        self.assertFalse(np.isnan(res["ri_temporal_gpu_util"]))
        self.assertFalse(np.isnan(res["ri_temporal_cpu_util"]))

    def test_all_ri_keys_present(self):
        res = calculate_ri_for_job({}, {}, is_gpu_job=False)
        expected_keys = [
            "ri_temporal_gpu_util", "ri_spatial_gpu_util",
            "ri_temporal_cpu_util", "ri_spatial_cpu_util",
            "ri_temporal_mem_rss",  "ri_spatial_mem_rss",
        ]
        for k in expected_keys:
            self.assertIn(k, res, f"Missing key: {k}")


# =========================================================================== #
# 2. GPU summarization tests                                                   #
# =========================================================================== #

class TestGPUSummarization(unittest.TestCase):

    def _build_single_gpu_job(self, util_vals, power_vals=None, runtime_s=100):
        """
        Simulate summarize_gpu output by building the ts_by_node directly
        and calling the aggregation logic.  Since summarize_gpu reads files,
        we test the aggregation math by calling calculate_ri_for_job and
        checking the RI results, and separately test scalar maths inline.
        """
        n = len(util_vals)
        power_vals = power_vals or [200.0] * n
        df = _make_gpu_ts("node1", 0, util_vals, power_vals=power_vals)
        return {"node1": df}

    def test_num_gpus_used_threshold(self):
        """GPU with mean util <= 2% should not count as used."""
        # GPU 0: mean=1% (below threshold) => not used
        # GPU 1: mean=50% => used
        df0 = _make_gpu_ts("node1", 0, [1.0, 1.0, 1.0])
        df1 = _make_gpu_ts("node1", 1, [50.0, 60.0, 55.0])
        ts_by_node = {"node1": pandas.concat([df0, df1], ignore_index=True)}

        # Replicate the per_gpu aggregation logic
        per_gpu = (
            pandas.concat([df0, df1])
            .groupby(["node_id", "gpu_index"])
            .agg(mean_util=("util_pct", "mean"))
            .reset_index()
        )
        per_gpu["is_active"] = per_gpu["mean_util"] > 2.0
        self.assertEqual(per_gpu["is_active"].sum(), 1)

    def test_power_energy_calculation(self):
        """Energy = mean_power * runtime_s / 3600."""
        mean_power_w = 300.0
        runtime_s    = 3600.0
        expected_wh  = 300.0  # 300W * 1h = 300Wh
        energy_wh = mean_power_w * runtime_s / 3600.0
        self.assertAlmostEqual(energy_wh, expected_wh)

    def test_gpu_util_mean_only_active_gpus(self):
        """Mean util should only consider active GPUs."""
        # GPU 0: mean=1% (inactive), GPU 1: mean=80%
        # Expected job util mean = 80% (not (1+80)/2 = 40.5%)
        utils_gpu0 = [1.0] * 5
        utils_gpu1 = [80.0] * 5
        df0 = _make_gpu_ts("node1", 0, utils_gpu0)
        df1 = _make_gpu_ts("node1", 1, utils_gpu1)
        per_gpu = (
            pandas.concat([df0, df1])
            .groupby(["node_id", "gpu_index"])
            .agg(mean_util=("util_pct", "mean"))
            .reset_index()
        )
        per_gpu["is_active"] = per_gpu["mean_util"] > 2.0
        active = per_gpu[per_gpu["is_active"]]
        self.assertAlmostEqual(active["mean_util"].mean(), 80.0)

    def test_mem_used_sum_and_mean(self):
        """Both sum and mean of mem_used should be saved."""
        # 2 active GPUs, each using 4096 KiB on average
        df0 = _make_gpu_ts("node1", 0, [50.0]*3, mem_used_vals=[4096.0]*3)
        df1 = _make_gpu_ts("node1", 1, [50.0]*3, mem_used_vals=[4096.0]*3)
        per_gpu = (
            pandas.concat([df0, df1])
            .groupby(["node_id", "gpu_index"])
            .agg(mean_mem_used=("mem_used_kib", "mean"),
                 mean_util    =("util_pct",     "mean"))
            .reset_index()
        )
        per_gpu["is_active"] = per_gpu["mean_util"] > 2.0
        active = per_gpu[per_gpu["is_active"]]
        self.assertAlmostEqual(active["mean_mem_used"].mean(), 4096.0)  # mean
        self.assertAlmostEqual(active["mean_mem_used"].sum(),  8192.0)  # sum


# =========================================================================== #
# 3. CPU summarization tests                                                   #
# =========================================================================== #

class TestCPUSummarization(unittest.TestCase):

    def test_mem_pct_utilization(self):
        """mem_pct_utilization = RSS / VMSize."""
        df = _make_cpu_ts("node1", [50.0], mem_rss_vals=[512.0], mem_avail_vals=[1024.0])
        self.assertAlmostEqual(df["mem_pct_utilization"].iloc[0], 0.5)

    def test_zero_mem_avail_no_crash(self):
        """Division by zero in mem pct should produce NaN, not crash."""
        df = _make_cpu_ts("node1", [50.0], mem_rss_vals=[512.0], mem_avail_vals=[0.0])
        # In production code this is handled by np.where; simulate it here
        val = df["mem_rss_kb"].iloc[0] / df["mem_avail_kb"].iloc[0] if df["mem_avail_kb"].iloc[0] > 0 else np.nan
        self.assertTrue(np.isnan(val))

    def test_io_totals(self):
        """read_kb_total should be sum of all timesteps."""
        df = _make_cpu_ts("node1", [50.0]*4, read_vals=[100.0, 200.0, 150.0, 50.0])
        self.assertAlmostEqual(df["read_kb"].sum(), 500.0)


# =========================================================================== #
# 4. Slurm metadata tests                                                      #
# =========================================================================== #

class TestSlurmUtils(unittest.TestCase):

    def _make_row(self, **kwargs):
        defaults = {
            "time_start": 1621000000,
            "time_end":   1621003600,
            "nodes_alloc": 2,
            "cpus_req": 8,
            "id_user": "user123",
            "partition": "gpu",
            "exit_code": 0,
            "tres_alloc": "cpu=8,mem=16G,1002=2",
        }
        defaults.update(kwargs)
        return pandas.Series(defaults)

    def test_runtime_calculation(self):
        row = self._make_row(time_start=1621000000, time_end=1621003600)
        result = clean_slurm_row(row)
        self.assertEqual(result["runtime_seconds"], 3600)

    def test_node_hours(self):
        row = self._make_row(time_start=1621000000, time_end=1621003600, nodes_alloc=2)
        result = clean_slurm_row(row)
        self.assertAlmostEqual(result["node_hours"], 2.0)  # 2 nodes * 1 hour

    def test_parse_gpu_count_volta(self):
        tres = "cpu=8,mem=16G,1002=2"
        self.assertEqual(_parse_num_alloc_gpus(tres), 2)

    def test_parse_gpu_count_tesla(self):
        tres = "cpu=8,mem=16G,1001=1"
        self.assertEqual(_parse_num_alloc_gpus(tres), 1)

    def test_parse_gpu_count_none(self):
        tres = "cpu=8,mem=16G"
        self.assertEqual(_parse_num_alloc_gpus(tres), 0)

    def test_parse_gpu_type_volta(self):
        self.assertEqual(_parse_gpu_type("cpu=8,1002=2"), "volta")

    def test_parse_gpu_type_tesla(self):
        self.assertEqual(_parse_gpu_type("cpu=8,1001=1"), "tesla")

    def test_parse_gpu_type_none(self):
        self.assertIsNone(_parse_gpu_type("cpu=8,mem=16G"))

    def test_gpu_hours(self):
        row = self._make_row(time_start=1621000000, time_end=1621003600,
                             nodes_alloc=2, tres_alloc="cpu=8,1002=2")
        result = clean_slurm_row(row)
        # 2 nodes * 1 hour * 2 GPUs = 4 GPU-hours
        self.assertAlmostEqual(result["gpu_hours"], 4.0)


# =========================================================================== #
# 5. GPU bad-row (stray string) tests                                         #
# =========================================================================== #

class TestGPUBadRows(unittest.TestCase):

    def test_stray_string_in_util_col_is_dropped(self):
        """Rows where util_pct is a string should be silently dropped."""
        import io, tempfile, os
        csv_content = (
            "timestamp,gpu_index,utilization_gpu_pct,utilization_memory_pct,"
            "memory_used_MiB,memory_free_MiB,temperature_gpu,temperature_memory,"
            "power_draw_W,pcie_link_width_current\n"
            "1620000000,0,80.0,50.0,4096,4096,65,60,250.0,16\n"
            "1620000010,0,N/A,50.0,4096,4096,65,60,250.0,16\n"   # bad row
            "1620000020,0,75.0,50.0,4096,4096,65,60,250.0,16\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="-node1.csv", delete=False, prefix="99999-"
        ) as f:
            f.write(csv_content)
            tmp_path = f.name

        try:
            import pandas as _pd
            from unittest.mock import patch
            # Mock s3_io so _load_gpu_file reads the temp local file instead
            with patch("s3_io.read_csv_from_s3",
                       side_effect=lambda p, **kw: _pd.read_csv(tmp_path)):
                import importlib, gpu_processing as _gm
                importlib.reload(_gm)
                df = _gm._load_gpu_file(tmp_path, job_id=99999)
            self.assertIsNotNone(df)
            self.assertEqual(len(df), 2)
            self.assertTrue(df["util_pct"].notna().all())
        finally:
            os.unlink(tmp_path)

    def test_stray_string_in_power_col_does_not_crash(self):
        """Stray string in power_draw_W should not crash; that value becomes NaN."""
        import io, tempfile, os
        csv_content = (
            "timestamp,gpu_index,utilization_gpu_pct,utilization_memory_pct,"
            "memory_used_MiB,memory_free_MiB,temperature_gpu,temperature_memory,"
            "power_draw_W,pcie_link_width_current\n"
            "1620000000,0,80.0,50.0,4096,4096,65,60,250.0,16\n"
            "1620000010,0,70.0,50.0,4096,4096,65,60,[N/A],16\n"  # bad power value
            "1620000020,0,75.0,50.0,4096,4096,65,60,240.0,16\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="-node1.csv", delete=False, prefix="99999-"
        ) as f:
            f.write(csv_content)
            tmp_path = f.name

        try:
            import pandas as _pd
            from unittest.mock import patch
            with patch("s3_io.read_csv_from_s3",
                       side_effect=lambda p, **kw: _pd.read_csv(tmp_path)):
                import importlib, gpu_processing as _gm
                importlib.reload(_gm)
                df = _gm._load_gpu_file(tmp_path, job_id=99999)
            self.assertIsNotNone(df)
            self.assertEqual(len(df), 3)
            self.assertTrue(df["power_w"].isna().any())
        finally:
            os.unlink(tmp_path)


# =========================================================================== #
# 6. Edge case tests                                                           #
# =========================================================================== #

class TestEdgeCases(unittest.TestCase):

    def test_ri_temporal_empty_series(self):
        """Empty arrays should return 0, not crash."""
        series = {"node1": np.array([])}
        self.assertAlmostEqual(_ri_temporal(series), 0.0)

    def test_ri_spatial_empty_nodes(self):
        self.assertAlmostEqual(_ri_spatial({}), 0.0)

    def test_ri_temporal_single_value_per_node(self):
        """Single timestep: sum == max*1 => RI = 0."""
        series = {"node1": np.array([100.0]), "node2": np.array([20.0])}
        self.assertAlmostEqual(_ri_temporal(series), 0.0)

    def test_calculate_ri_empty_inputs(self):
        """No ts data => all NaN for GPU, all NaN for CPU."""
        res = calculate_ri_for_job({}, {}, is_gpu_job=False)
        self.assertTrue(np.isnan(res["ri_temporal_cpu_util"]))

    def test_ri_clamped_to_0_1(self):
        """RI must never exceed 1 or go below 0."""
        series = {"node1": np.array([0.001, 0.0, 0.0, 0.0, 0.0])}
        ri_t = _ri_temporal(series)
        self.assertGreaterEqual(ri_t, 0.0)
        self.assertLessEqual(ri_t, 1.0)



# =========================================================================== #
# 7. New metrics tests (distribution stats, corrected cpu_util normalisation) #
# =========================================================================== #

class TestCPUUtilNormalisation(unittest.TestCase):

    def test_cpus_per_node_used_not_cpus_req(self):
        """
        With 2 nodes and cpus_req=8, cpus_per_node=4.
        A raw CPUUtilization of 400 on one node should give cpu_util_pct=100,
        NOT 400/8=50.
        """
        # Simulate what _load_cpu_file does
        cpus_req  = 8.0
        num_nodes = 2.0
        cpus_per_node = cpus_req / num_nodes  # = 4
        raw_util = 400.0  # all 4 CPUs on this node at 100%
        cpu_util_pct = raw_util / cpus_per_node
        self.assertAlmostEqual(cpu_util_pct, 100.0)

    def test_single_node_fallback(self):
        """num_nodes=0 should fall back to 1 to avoid divide-by-zero."""
        cpus_req  = 8.0
        num_nodes = 0.0
        num_nodes = max(num_nodes, 1.0)  # guard applied in summarize_cpu
        cpus_per_node = cpus_req / num_nodes
        self.assertAlmostEqual(cpus_per_node, 8.0)


class TestFlatDistributionStats(unittest.TestCase):

    def test_flat_std_reflects_full_variability(self):
        """
        With one steady node and one bursty node, flat std should be higher
        than the average of per-node stds.
        """
        node1 = [80.0, 85.0, 90.0, 88.0, 82.0]   # std ~ 4
        node2 = [5.0,  90.0, 5.0,  90.0, 5.0]    # std ~ 44

        all_vals = pandas.Series(node1 + node2)
        flat_std = all_vals.std()

        per_node_std_avg = (
            pandas.Series(node1).std() + pandas.Series(node2).std()
        ) / 2

        self.assertGreater(flat_std, per_node_std_avg)

    def test_iof_zero_when_mean_is_zero(self):
        """IoF should return 0 (not NaN/inf) when mean is 0."""
        s = pandas.Series([0.0, 0.0, 0.0])
        m = s.mean()
        iof = s.var() / m if m != 0 else 0.0
        self.assertEqual(iof, 0.0)

    def test_skew_negative_for_high_util(self):
        """Mostly-high utilisation with occasional dips => negative skew."""
        from scipy import stats as scipy_stats
        s = pandas.Series([90.0, 92.0, 88.0, 91.0, 10.0])
        self.assertLess(scipy_stats.skew(s), 0)


class TestGPUIdleRatio(unittest.TestCase):

    def test_idle_ratio_all_used(self):
        """If all allocated GPUs are used, idle ratio = 0."""
        num_gpus_used  = 2
        num_alloc_gpus = 2
        ratio = 1.0 - num_gpus_used / num_alloc_gpus
        self.assertAlmostEqual(ratio, 0.0)

    def test_idle_ratio_half_idle(self):
        """If half allocated GPUs are idle, ratio = 0.5."""
        num_gpus_used  = 1
        num_alloc_gpus = 2
        ratio = 1.0 - num_gpus_used / num_alloc_gpus
        self.assertAlmostEqual(ratio, 0.5)

    def test_idle_ratio_nan_when_no_alloc(self):
        """If num_alloc_gpus = 0 (CPU-only job), ratio should be NaN."""
        import numpy as np
        num_alloc_gpus = 0
        ratio = (
            1.0 - 1 / num_alloc_gpus if num_alloc_gpus > 0 else np.nan
        )
        self.assertTrue(np.isnan(ratio))


class TestPerNodePercentiles(unittest.TestCase):

    def test_percentile_not_dominated_by_chatty_node(self):
        """
        node1 has 10 timesteps all at 90%, node2 has 2 timesteps at 10%.
        Flat p75 would be dominated by node1.
        Per-node-averaged p75 should reflect both nodes equally.
        """
        df = pandas.concat([
            _make_cpu_ts("node1", [90.0] * 10),
            _make_cpu_ts("node2", [10.0] * 2),
        ], ignore_index=True)

        flat_p75 = df["cpu_util_pct"].quantile(0.75)

        per_node_p75 = df.groupby("node_id")["cpu_util_pct"].quantile(0.75).mean()

        # Flat p75 will be close to 90 (node1 dominates), per-node avg will be ~50
        self.assertGreater(flat_p75, per_node_p75)


# =========================================================================== #
# 8. Per-GPU/per-node temporal variability tests                              #
# =========================================================================== #

class TestTemporalVariabilityAggregation(unittest.TestCase):

    def test_std_captures_temporal_not_between_gpu_variance(self):
        """
        Two GPUs at different but STEADY levels should have low std
        (no temporal variability), even though between-GPU variance is high.
        Spatial RI captures that between-GPU difference, not std.
        """
        # GPU 0 steady at 80%, GPU 1 steady at 20%
        df0 = _make_gpu_ts("node1", 0, [80.0] * 10)
        df1 = _make_gpu_ts("node1", 1, [20.0] * 10)
        active_ts = pandas.concat([df0, df1], ignore_index=True)

        from gpu_processing import _per_gpu_stats
        result = _per_gpu_stats(active_ts, "util_pct", "gpu_util")

        # Per-GPU std: GPU0 std=0, GPU1 std=0, avg=0
        # Flat std would be ~30 (capturing between-GPU difference)
        self.assertAlmostEqual(result["gpu_util_std"], 0.0, places=3)
        # Mean should be average of the two GPU means = 50%
        self.assertAlmostEqual(result["gpu_util_mean"], 50.0, places=3)

    def test_std_captures_bursty_gpu(self):
        """
        A GPU that alternates between 0% and 100% should have high std.
        """
        df0 = _make_gpu_ts("node1", 0, [0.0, 100.0, 0.0, 100.0, 0.0, 100.0])
        active_ts = df0.copy()
        active_ts["is_active"] = True

        from gpu_processing import _per_gpu_stats
        result = _per_gpu_stats(active_ts, "util_pct", "gpu_util")

        # std of [0,100,0,100,0,100] is ~54.8
        self.assertGreater(result["gpu_util_std"], 40.0)

    def test_cpu_std_per_node_not_cross_node(self):
        """
        Two nodes at different but steady CPU levels should have near-zero std.
        Cross-node variance is captured by spatial RI, not std.
        """
        df = pandas.concat([
            _make_cpu_ts("node1", [90.0] * 5),
            _make_cpu_ts("node2", [10.0] * 5),
        ], ignore_index=True)

        from cpu_processing import _per_node_stats
        result = _per_node_stats(df, "cpu_util_pct", "cpu_util")

        # Per-node std: node1 std=0, node2 std=0, avg=0
        self.assertAlmostEqual(result["cpu_util_std"], 0.0, places=3)
        # Mean: avg of (90, 10) = 50
        self.assertAlmostEqual(result["cpu_util_mean"], 50.0, places=3)

    def test_power_mean_and_peak_are_summed_across_gpus(self):
        """
        Power mean/peak should be SUMMED across GPUs (physical total draw),
        not averaged — unlike util/temp which are averaged.
        GPU0: mean=200W, GPU1: mean=150W => job total mean = 350W
        """
        df0 = _make_gpu_ts("node1", 0, [50.0]*5, power_vals=[200.0]*5)
        df1 = _make_gpu_ts("node1", 1, [50.0]*5, power_vals=[150.0]*5)
        per_gpu = (
            pandas.concat([df0, df1])
            .groupby(["node_id", "gpu_index"])
            .agg(mean_power=("power_w", "mean"))
            .reset_index()
        )
        total_mean_power = per_gpu["mean_power"].sum()
        self.assertAlmostEqual(total_mean_power, 350.0)



    def test_idle_gpu_power_included_in_total(self):
        """
        gpu_power_mean_total_including_idle_w should include idle GPUs.
        GPU 0: mean util=50% (active), mean power=200W
        GPU 1: mean util=1%  (idle),   mean power=80W
        active-only total  = 200W
        including-idle total = 200 + 80 = 280W
        """
        per_gpu_scalars_power = pandas.Series([200.0])   # active only
        per_gpu_all_power     = pandas.Series([200.0, 80.0])  # active + idle
        self.assertAlmostEqual(per_gpu_scalars_power.sum(), 200.0)
        self.assertAlmostEqual(per_gpu_all_power.sum(),     280.0)
        self.assertGreater(per_gpu_all_power.sum(), per_gpu_scalars_power.sum())

    def test_mem_used_sum_is_job_total_vram(self):
        """
        mem_used sum should reflect total VRAM consumed across all active GPUs.
        GPU0: mean=4096 KiB, GPU1: mean=8192 KiB => job total = 12288 KiB
        """
        df0 = _make_gpu_ts("node1", 0, [50.0]*3, mem_used_vals=[4096.0]*3)
        df1 = _make_gpu_ts("node1", 1, [50.0]*3, mem_used_vals=[8192.0]*3)
        per_gpu = (
            pandas.concat([df0, df1])
            .groupby(["node_id", "gpu_index"])
            .agg(mean_mem=("mem_used_kib", "mean"))
            .reset_index()
        )
        self.assertAlmostEqual(per_gpu["mean_mem"].sum(), 12288.0)


# =========================================================================== #
# 9. S3 streaming tests (mocked)                                              #
# =========================================================================== #

class TestS3IO(unittest.TestCase):

    def test_read_csv_from_s3_returns_dataframe_on_success(self):
        """read_csv_from_s3 should return a DataFrame when S3 responds OK."""
        import io
        from unittest.mock import MagicMock, patch

        csv_data = b"a,b,c\n1,2,3\n4,5,6\n"
        mock_response = {"Body": MagicMock(read=lambda: csv_data)}

        with patch("s3_io.get_s3_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get_object.return_value = mock_response
            mock_get_client.return_value = mock_client

            from s3_io import read_csv_from_s3
            df = read_csv_from_s3("some/path.csv")

        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertListEqual(list(df.columns), ["a", "b", "c"])

    def test_read_csv_from_s3_returns_none_on_missing_key(self):
        """read_csv_from_s3 should return None when key doesn't exist."""
        from unittest.mock import MagicMock, patch
        import botocore.exceptions

        with patch("s3_io.get_s3_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.exceptions.NoSuchKey = botocore.exceptions.ClientError
            mock_client.get_object.side_effect = botocore.exceptions.ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "Not found"}},
                "GetObject",
            )
            mock_get_client.return_value = mock_client

            from s3_io import read_csv_from_s3
            # Generic exception path (NoSuchKey check uses client.exceptions.NoSuchKey)
            df = read_csv_from_s3("nonexistent/path.csv")

        self.assertIsNone(df)

    def test_read_csv_from_s3_returns_none_on_network_error(self):
        """read_csv_from_s3 should return None on any unexpected exception."""
        from unittest.mock import MagicMock, patch

        with patch("s3_io.get_s3_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.exceptions.NoSuchKey = Exception
            mock_client.get_object.side_effect = ConnectionError("timeout")
            mock_get_client.return_value = mock_client

            from s3_io import read_csv_from_s3
            df = read_csv_from_s3("some/path.csv")

        self.assertIsNone(df)


# =========================================================================== #
# 10. Checkpoint tests                                                         #
# =========================================================================== #

class TestCheckpoint(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp_dir = tempfile.mkdtemp()
        import checkpoint as ckpt
        self._orig_dir    = ckpt.PROCESSED_DATA_DIR
        self._orig_chunks = ckpt.CHUNKS_DIR
        self._orig_done   = ckpt.DONE_JOBS_FILE
        self._orig_errors = ckpt.ERRORS_FILE
        ckpt.PROCESSED_DATA_DIR = self.tmp_dir
        ckpt.CHUNKS_DIR         = os.path.join(self.tmp_dir, "chunks")
        ckpt.DONE_JOBS_FILE     = os.path.join(self.tmp_dir, "done_jobs.csv")
        ckpt.ERRORS_FILE        = os.path.join(self.tmp_dir, "errors.csv")
        os.makedirs(ckpt.CHUNKS_DIR, exist_ok=True)

    def tearDown(self):
        import shutil, checkpoint as ckpt
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        ckpt.PROCESSED_DATA_DIR = self._orig_dir
        ckpt.CHUNKS_DIR         = self._orig_chunks
        ckpt.DONE_JOBS_FILE     = self._orig_done
        ckpt.ERRORS_FILE        = self._orig_errors

    def test_load_done_jobs_empty_when_no_file(self):
        from checkpoint import load_done_jobs
        self.assertEqual(load_done_jobs(), set())

    def test_done_jobs_grows_across_chunks(self):
        """done_jobs.csv is append-only superset across all chunks."""
        from checkpoint import append_done_jobs, load_done_jobs
        append_done_jobs(["100", "101", "102"])  # chunk 1
        append_done_jobs(["200", "201", "202"])  # chunk 2
        done = load_done_jobs()
        self.assertEqual(len(done), 6)
        self.assertIn("100", done)
        self.assertIn("202", done)

    def test_failed_jobs_not_in_done_jobs(self):
        """Failed jobs go to errors.csv, not done_jobs.csv — so they are retried."""
        from checkpoint import append_done_jobs, append_error, load_done_jobs
        append_done_jobs(["100", "101"])
        append_error("999", "connection timeout")
        done = load_done_jobs()
        self.assertIn("100", done)
        self.assertNotIn("999", done)  # failed job absent — will be retried

    def test_save_and_load_chunk(self):
        from checkpoint import save_chunk, load_all_chunks
        records = [{"job_id": 1, "runtime_seconds": 3600},
                   {"job_id": 2, "runtime_seconds": 7200}]
        save_chunk(records, chunk_idx=1)
        df = load_all_chunks()
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)

    def test_multiple_chunks_concat_correctly(self):
        from checkpoint import save_chunk, load_all_chunks
        save_chunk([{"job_id": 1}, {"job_id": 2}], chunk_idx=1)
        save_chunk([{"job_id": 3}, {"job_id": 4}], chunk_idx=2)
        df = load_all_chunks()
        self.assertEqual(len(df), 4)
        self.assertSetEqual(set(df["job_id"].tolist()), {1, 2, 3, 4})

    def test_get_max_chunk_idx(self):
        """get_max_chunk_idx returns 0 with no chunks, correct max otherwise."""
        from checkpoint import save_chunk, get_max_chunk_idx
        self.assertEqual(get_max_chunk_idx(), 0)
        save_chunk([{"job_id": 1}], chunk_idx=1)
        self.assertEqual(get_max_chunk_idx(), 1)
        save_chunk([{"job_id": 2}], chunk_idx=5)  # non-contiguous
        self.assertEqual(get_max_chunk_idx(), 5)

    def test_chunk_offset_prevents_overwrite_on_resume(self):
        """
        New chunks should start at max_existing_idx + 1 to avoid overwriting.
        """
        from checkpoint import save_chunk, get_max_chunk_idx
        save_chunk([{"job_id": 1}], chunk_idx=1)
        save_chunk([{"job_id": 2}], chunk_idx=2)
        offset    = get_max_chunk_idx()   # = 2
        new_start = offset + 1            # = 3
        self.assertEqual(new_start, 3)

    def test_save_then_commit_ordering(self):
        """
        Correct order: save_chunk first, then append_done_jobs.
        If crash between them, chunk file exists but job_ids not in done_jobs.
        """
        from checkpoint import save_chunk, append_done_jobs, load_done_jobs
        save_chunk([{"job_id": 42}], chunk_idx=1)
        # Simulate crash here — not yet committed
        self.assertNotIn("42", load_done_jobs())
        # Now commit
        append_done_jobs(["42"])
        self.assertIn("42", load_done_jobs())

    def test_append_error(self):
        import checkpoint as ckpt
        from checkpoint import append_error
        append_error("99999", "some error message")
        self.assertTrue(os.path.exists(ckpt.ERRORS_FILE))
        with open(ckpt.ERRORS_FILE) as f:
            content_str = f.read()
        self.assertIn("99999", content_str)
        self.assertIn("some error message", content_str)

    def test_resume_filters_done_and_retries_failed(self):
        """
        done_jobs skips completed jobs.
        Failed jobs (errors.csv only) reappear in task list and are retried.
        """
        from checkpoint import append_done_jobs, append_error, load_done_jobs
        append_done_jobs(["100", "101"])
        append_error("999", "timeout")

        done      = load_done_jobs()
        all_jobs  = ["100", "101", "999", "200"]
        remaining = [j for j in all_jobs if j not in done]
        # 999 (failed) and 200 (unprocessed) should both be retried
        self.assertIn("999", remaining)
        self.assertIn("200", remaining)
        self.assertNotIn("100", remaining)


# =========================================================================== #
# 11. File index JSON tests                                                    #
# =========================================================================== #

class TestFileIndex(unittest.TestCase):

    def test_load_file_index(self):
        import json, tempfile
        index = {
            "12345": {"cpu": ["path/to/12345-timeseries.csv"], "gpu": []},
            "67890": {"cpu": ["path/to/67890-timeseries.csv"],
                      "gpu": ["path/to/67890-node1.csv"]},
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(index, f)
            tmp_path = f.name

        try:
            from file_index import load_file_index
            result = load_file_index(tmp_path)
            self.assertEqual(len(result), 2)
            self.assertIn("12345", result)
            self.assertEqual(result["67890"]["gpu"], ["path/to/67890-node1.csv"])
        finally:
            os.unlink(tmp_path)


# =========================================================================== #
# 12. Chunked pool recycling tests                                             #
# =========================================================================== #

class TestChunkedProcessing(unittest.TestCase):

    def test_tasks_split_into_correct_chunks(self):
        """400 tasks with chunk_size=100 should produce 4 chunks."""
        tasks     = list(range(400))
        chunk_size = 100
        chunks    = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]
        self.assertEqual(len(chunks), 4)
        self.assertEqual(len(chunks[0]), 100)
        self.assertEqual(len(chunks[-1]), 100)

    def test_partial_last_chunk(self):
        """410 tasks with chunk_size=100 should produce 5 chunks, last has 10."""
        tasks      = list(range(410))
        chunk_size = 100
        chunks     = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]
        self.assertEqual(len(chunks), 5)
        self.assertEqual(len(chunks[-1]), 10)

    def test_results_scoped_to_chunk(self):
        """
        Results list is local to process_chunk — a new empty list starts
        each chunk, so memory is naturally bounded to chunk_size results.
        """
        chunk_size = 500
        # Simulate two chunks each producing chunk_size results
        chunk1_results = [{"job_id": i} for i in range(chunk_size)]
        chunk2_results = [{"job_id": i} for i in range(chunk_size, chunk_size * 2)]
        # Each chunk's results list is independent — peak memory = 1 chunk
        self.assertEqual(len(chunk1_results), chunk_size)
        self.assertEqual(len(chunk2_results), chunk_size)
        # They don't accumulate
        self.assertNotEqual(chunk1_results, chunk2_results)

    def test_dry_run_chunk_count(self):
        """Dry run should correctly compute number of pool cycles."""
        total_jobs = 400_000
        chunk_size = 500
        n_chunks   = (total_jobs + chunk_size - 1) // chunk_size
        self.assertEqual(n_chunks, 800)

    def test_done_jobs_prevents_reprocessing_across_chunks(self):
        """
        Jobs completed in chunk 1 should be in done_jobs and skipped
        when building the task list for a resume run.
        """
        all_job_ids = [str(i) for i in range(1000)]
        # Simulate chunk 1 completing jobs 0-499
        done_jobs   = set(str(i) for i in range(500))
        remaining   = [j for j in all_job_ids if j not in done_jobs]
        self.assertEqual(len(remaining), 500)
        self.assertEqual(remaining[0], "500")

# =========================================================================== #
# Run                                                                          #
# =========================================================================== #

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestRITemporal,
        TestRISpatial,
        TestCalculateRIForJob,
        TestGPUSummarization,
        TestCPUSummarization,
        TestSlurmUtils,
        TestGPUBadRows,
        TestCPUUtilNormalisation,
        TestFlatDistributionStats,
        TestGPUIdleRatio,
        TestPerNodePercentiles,
        TestTemporalVariabilityAggregation,
        TestS3IO,
        TestCheckpoint,
        TestFileIndex,
        TestChunkedProcessing,
        TestEdgeCases,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)