"""Read-only deployment diagnostics; does not load model weights."""
import json,platform,shutil,sys
from pathlib import Path
import psutil
import openvino as ov
ROOT=Path(__file__).resolve().parent
core=ov.Core();devices=core.available_devices
print(json.dumps({'python':sys.version,'os':platform.platform(),
 'ram_gib':round(psutil.virtual_memory().total/2**30,2),
 'available_ram_gib':round(psutil.virtual_memory().available/2**30,2),
 'openvino_devices':{d:core.get_property(d,'FULL_DEVICE_NAME') for d in devices},
 'ffmpeg':shutil.which('ffmpeg'),
 'source_model_present':(ROOT/'models/trelis/model.safetensors.index.json').is_file(),
 'intel_export_present':(ROOT/'models/trelis-openvino-fp16/openvino_decoder_model.bin').is_file()},indent=2))
