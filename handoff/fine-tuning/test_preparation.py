import json, tempfile, unittest
from pathlib import Path
from common import normalize,errors,validate_splits,CONFIG

class PreparationTests(unittest.TestCase):
    def test_hindi_marks_survive(self):
        self.assertEqual(normalize('किताब, नहीं!'), 'किताब नहीं')
        self.assertEqual(errors('किताब नहीं','कतब नही')[0],2)
    def test_corpus_error_counts(self):
        self.assertEqual(errors('a b c','a c'),(1,3))
        self.assertEqual(errors('','hallucinated words'),(2,0))
    def test_group_leakage_is_rejected(self):
        from unittest.mock import patch
        rows=[dict(id='a',group_id='same',split='train',text='hello',reviewed=True,license_verified=True,audio=__file__),
              dict(id='b',group_id='same',split='test',text='hello',reviewed=True,license_verified=True,audio=__file__)]
        with patch('common.audio_array'), self.assertRaisesRegex(ValueError,'Group leakage'):validate_splits(rows)
    def test_unreviewed_data_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'reviewed'):validate_splits([dict(split='train',reviewed=False,text='draft')])
    def test_publisher_exception_never_applies_to_test(self):
        row=dict(split='test',reviewed=False,text='draft',label_origin='publisher',annotation_status='publisher_unverified',source_license={'source_url':'https://example.com'})
        with self.assertRaisesRegex(ValueError,'reviewed'):validate_splits([row],allow_publisher_labels=True)
    def test_encoder_is_excluded(self):
        import re
        self.assertIsNone(re.fullmatch(CONFIG['target_modules'],'model.encoder.layers.0.self_attn.q_proj'))
        self.assertIsNotNone(re.fullmatch(CONFIG['target_modules'],'model.decoder.layers.31.encoder_attn.v_proj'))

if __name__=='__main__':unittest.main()
