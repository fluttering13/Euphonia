"""Build an offline listening comparison from actual benchmark results."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def main():
    labels = {'faster': 'Qwen3-TTS 0.6B 加速版',
              'qwen_streaming': 'Qwen3-TTS-streaming',
              'f5': 'F5-TTS v1（16 步，PyTorch CUDA）',
              'zipvoice': 'ZipVoice-Distill（4 步，PyTorch CUDA）'}
    blocks = []
    for backend, label in labels.items():
        path = DATA / f'comparison-{backend}.json'
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        rows = []
        for row in data.get('runs', []):
            if row['label'] in ('first', 'first_request'):
                continue
            file = Path(row['file'])
            if not file.is_absolute():
                file = ROOT / file
            relative = file.relative_to(DATA).as_posix()
            text = row.get('text', 'The morning sky looks beautiful. Shall we explore the forest together?')
            elapsed = row.get('request_seconds', row['total_seconds'])
            rows.append(f'<tr><td>{html.escape(text)}</td><td>{elapsed:.3f} 秒</td>'
                        f'<td>{row["audio_seconds"]:.2f} 秒</td><td><audio controls preload="none" src="{html.escape(relative)}"></audio></td></tr>')
        error = f'<p>{html.escape(data["error"])}</p>' if data.get('error') else ''
        blocks.append(f'<section><h2>{label}</h2>{error}<table><thead><tr><th>英文台詞</th><th>整段生成</th><th>音訊長度</th><th>試聽</th></tr></thead><tbody>{"".join(rows)}</tbody></table></section>')
    page = '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>Euphonia 模型比較</title>
<style>body{font:16px system-ui;background:#101820;color:#e3edf0;max-width:1150px;margin:40px auto;padding:0 20px}h1,h2{color:#69e0c0}section{margin:35px 0}table{width:100%;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid #354a58;padding:15px 10px}td:first-child{max-width:460px}audio{width:270px}p{line-height:1.7;color:#b9cbd2}</style>
<h1>Euphonia 英文模型比較</h1>
<p>同一份 Rhiannon 採樣；表列暖機後的完整音訊生成與 WAV 寫入時間，不含 OCR、人工框選及實際播放。各模型生成的語速與音訊長度可能不同。尚未控制遊戲負載，三次中句測試不足以代表長期穩定性。</p>
<p>參考聲音：<audio controls preload="none" src="voices/d474ca0976b5497687bd18e338ff0b18/reference.wav"></audio></p>
<p>請比較音色相似度、自然度、英文發音及漏字。ASR 檢查不能替代音色試聽。</p>
''' + ''.join(blocks) + '</html>'
    (DATA / 'model-comparison.html').write_text(page, encoding='utf-8')
    print(DATA / 'model-comparison.html')


if __name__ == '__main__':
    main()
