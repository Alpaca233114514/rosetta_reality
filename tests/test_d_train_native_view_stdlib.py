"""Whitelist, complete chunk accounting and logical identity checks; no ML imports."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import copy
from rosetta_reality.diagnostics.d_train_native_view import inventory,TRAIN,PROTECTED,EpisodeColumnProxy
from rosetta_reality.diagnostics import d_train_native_view as view


class DNativeInventoryTests(unittest.TestCase):
    def test_both_complete_original_and_full_chunk_inventories_are_preserved(self):
        result=inventory({ep:500 for ep in TRAIN})
        self.assertEqual(result["input_frames"],20000)
        self.assertEqual(result["full_chunk_input_frames"],18040)
        self.assertEqual(result["valid_target_slots"],951000)
        self.assertEqual(result["padding_slots"],49000)
        self.assertEqual(result["full_chunk_valid_target_slots"],902000)
        self.assertEqual([row["episode"] for row in result["inputs"][::500]],list(TRAIN))

    def test_padding_uses_original_final_action_not_a_new_target(self):
        result=inventory({49:500})
        self.assertEqual(result["inputs"][450]["valid_target_slots"],50)
        self.assertEqual(result["inputs"][451]["valid_target_slots"],49)
        self.assertEqual(result["inputs"][499]["valid_target_slots"],1)
        self.assertEqual(result["full_chunk_inputs"][-1]["frame"],450)

    def test_protected_rows_rejected_instead_of_filtered(self):
        for ep in PROTECTED:
            with self.subTest(ep=ep),self.assertRaises(ValueError):inventory({ep:500})

    def test_episode_lengths_cannot_be_faked(self):
        with self.assertRaises(ValueError):inventory({49:499})

    def test_native_column_mapping_retains_original_training_identities(self):
        columns={"episode_index":[0,0,1,1],"frame_index":[0,1,0,1]}
        proxy=EpisodeColumnProxy(columns,{0:4,1:49})
        self.assertEqual(proxy["episode_index"],[4,4,49,49])
        self.assertEqual(proxy["frame_index"],[0,1,0,1])
        self.assertEqual(columns["episode_index"],[0,0,1,1])

    def test_actual_factory_requires_full_pixel_label_and_statistics_acceptance(self):
        manifest={"all_40_episodes_present":True,"train_original_episodes":list(TRAIN),
                  "normalization_sha256":view.NORMALIZATION_SHA}
        proof={"status":"passed","native_read_frames":20000,"original_episodes":list(TRAIN),
               "view_manifest_sha256":"manifest","normalization_sha256":view.NORMALIZATION_SHA,
               "label_parity":"all_selected_states_and_chunk_actions_exact",
               "pixel_parity":"all_selected_frames_exact"}
        def hashed(path):
            return {"view-manifest.json":"manifest","native-read-verification.json":"proof",
                    "normalization.json":view.NORMALIZATION_SHA}[path.name]
        for field,value in (("label_parity",None),("pixel_parity",None),("native_read_frames",19999),
                            ("original_episodes",list(TRAIN[:-1]))):
            altered=copy.deepcopy(proof);altered[field]=value
            module=SimpleNamespace(make_train_eval_datasets=lambda _:None)
            original=module.make_train_eval_datasets
            with self.subTest(field=field),patch.object(view,"read",side_effect=[manifest,altered]),patch.object(view,"sha",side_effect=hashed):
                with self.assertRaises(ValueError):
                    view.install_native_D_factory(module,".",view_manifest_sha256="manifest",native_read_sha256="proof",delta_timestamps={})
            self.assertIs(module.make_train_eval_datasets,original)
        with patch.object(view,"read",side_effect=[manifest,proof]),patch.object(view,"sha",side_effect=lambda p:"changed" if p.name=="normalization.json" else hashed(p)):
            with self.assertRaises(ValueError):view.accepted_training_view(".","manifest",native_read_sha256="proof")

    def test_actual_factory_acceptance_preserves_original_and_restores_exact_binding(self):
        manifest={"all_40_episodes_present":True,"train_original_episodes":list(TRAIN),"normalization_sha256":view.NORMALIZATION_SHA}
        proof={"status":"passed","native_read_frames":20000,"original_episodes":list(TRAIN),"view_manifest_sha256":"manifest",
               "normalization_sha256":view.NORMALIZATION_SHA,"label_parity":"all_selected_states_and_chunk_actions_exact","pixel_parity":"all_selected_frames_exact"}
        module=SimpleNamespace(make_train_eval_datasets=lambda _:None);original=module.make_train_eval_datasets
        with patch.object(view,"read",side_effect=[manifest,proof]),patch.object(view,"sha",side_effect=lambda p:{"view-manifest.json":"manifest","native-read-verification.json":"proof","normalization.json":view.NORMALIZATION_SHA}[p.name]):
            restore=view.install_native_D_factory(module,".",view_manifest_sha256="manifest",native_read_sha256="proof",delta_timestamps={})
        self.assertIsNot(module.make_train_eval_datasets,original)
        restore();self.assertIs(module.make_train_eval_datasets,original)

    def test_metadata_lookup_removes_custom_image_transform_without_altering_native_rows(self):
        class Metadata:
            def with_format(self,format):
                self.format=format;return self
            def __getitem__(self,key):return {"episode_index":[0,1],"frame_index":[0,499]}[key]
        class Native:
            column_names=["episode_index","frame_index","observation.images.top","action"]
            def select_columns(self,fields):
                self.fields=fields;self.meta=Metadata();return self.meta
            def __getitem__(self,key):raise AssertionError("Metadata must not invoke full native image transform")
        native=Native();proxy=EpisodeColumnProxy(native,{0:4,1:49})
        self.assertEqual(native.fields,["episode_index","frame_index"])
        self.assertIsNone(native.meta.format)
        self.assertEqual(proxy["episode_index"],[4,49]);self.assertEqual(proxy["frame_index"],[0,499])


if __name__=="__main__":unittest.main()
