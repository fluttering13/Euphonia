"""Compare OCR configurations without changing the application's configuration."""
import gc
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
from euphonia.core import DATA

image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / 'sample-tests/rhiannon/ocr-input.png'
report_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DATA / 'ocr-profile.json'
image = np.asarray(Image.open(image_path).convert('RGB'))[:, :, ::-1].copy()
cases = [('default', {}),
         ('threads_2', {'intra_op_num_threads': 2, 'inter_op_num_threads': 1}),
         ('threads_4', {'intra_op_num_threads': 4, 'inter_op_num_threads': 1}),
         ('threads_4_max_960', {'intra_op_num_threads': 4, 'inter_op_num_threads': 1,
                              'det_limit_type': 'max', 'det_limit_side_len': 960})]
report = []
for name, settings in cases:
    engine = RapidOCR(**settings)
    seconds = []
    for _ in range(4):
        start = time.perf_counter()
        output, timings = engine(image)
        seconds.append(time.perf_counter() - start)
    row = dict(name=name, configuration=settings, seconds=seconds, warm_median=statistics.median(seconds[1:]),
               text='\n'.join(item[1] for item in (output or [])), internal_stage_seconds=timings)
    report.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    del engine
    gc.collect()
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
