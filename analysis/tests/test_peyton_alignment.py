from django.test import SimpleTestCase

from peyton_original.DataCleaning import config as peyton_config
from ml_partner_adapters import config_bridge
from ml_partner_adapters import model_configs as adapter_model_configs
from analysis.ml_core import cleaning as engine_cleaning
from analysis.ml_core.models import MODEL_REGISTRY


class PeytonConfigAlignmentTests(SimpleTestCase):
    def test_config_bridge_matches_peyton_config(self):
        self.assertEqual(
            config_bridge.UNKNOWN_THRESHOLD,
            peyton_config.UNKNOWN_THRESHOLD,
        )
        self.assertEqual(
            config_bridge.YES_NO_THRESHOLD,
            peyton_config.YES_NO_THRESHOLD,
        )
        self.assertEqual(
            set(config_bridge.UNKNOWN_STRINGS),
            set(peyton_config.UNKNOWN_STRINGS),
        )

    def test_engine_config_imports_via_adapter(self):
        # The engine should be importing config values from the adapter
        # so that updating Peyton's repo only requires adapter changes.
        self.assertEqual(
            engine_cleaning.UNKNOWN_THRESHOLD,
            config_bridge.UNKNOWN_THRESHOLD,
        )
        self.assertEqual(
            engine_cleaning.YES_NO_THRESHOLD,
            config_bridge.YES_NO_THRESHOLD,
        )

        engine_unknowns = set(engine_cleaning.DEFAULT_UNKNOWN_STRINGS)
        adapter_unknowns = set(config_bridge.UNKNOWN_STRINGS)
        # Engine defaults should at least include all canonical Peyton strings.
        self.assertTrue(adapter_unknowns <= engine_unknowns)


class PeytonModelDefaultsAlignmentTests(SimpleTestCase):
    def test_decision_tree_defaults_align_with_adapter(self):
        spec = MODEL_REGISTRY["crash_severity_risk_v1"]
        self.assertEqual(
            spec.default_model_params,
            adapter_model_configs.DECISION_TREE_BASE_PARAMS,
        )

    def test_ebm_defaults_align_with_adapter(self):
        spec = MODEL_REGISTRY["ebm_v1"]
        self.assertEqual(
            spec.default_model_params,
            adapter_model_configs.EBM_BASE_PARAMS,
        )

    def test_mrf_defaults_align_with_adapter(self):
        spec = MODEL_REGISTRY["mrf_v1"]
        self.assertEqual(
            spec.default_model_params,
            adapter_model_configs.MRF_BASE_PARAMS,
        )

    def test_xgb_defaults_align_with_adapter(self):
        spec = MODEL_REGISTRY["xgb_v1"]
        self.assertEqual(
            spec.default_model_params,
            adapter_model_configs.XGB_BASE_PARAMS,
        )
