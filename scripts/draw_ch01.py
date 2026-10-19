#!/usr/bin/env python3
"""Deterministic original SVG schematics; no measured data or extra dependencies."""
from pathlib import Path
from html import escape
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets' / 'images'

def diagram(name, title, boxes, footer):
    elements = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="300" viewBox="0 0 960 300" role="img">',
        f'<title>{escape(title)}</title>', '<rect width="960" height="300" fill="#f7f8fc"/>',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="#44536d"/></marker></defs>',
        '<g font-family="sans-serif" fill="#18243b">',
        f'<text x="35" y="45" font-size="24" font-weight="bold">{escape(title)}</text>']
    for i, (head, detail) in enumerate(boxes):
        x=35+i*235
        elements.extend([f'<rect x="{x}" y="90" width="210" height="110" rx="12" fill="white" stroke="#637cb2"/>',
          f'<text x="{x+14}" y="125" font-size="19">{escape(head)}</text>',
          f'<text x="{x+14}" y="165" font-size="15">{escape(detail)}</text>'])
        if i < len(boxes)-1:
            elements.append(f'<path d="M{x+212} 145 L{x+232} 145" stroke="#44536d" stroke-width="2" marker-end="url(#arrow)"/>')
    elements.extend([f'<text x="35" y="250" font-size="17">{escape(footer)}</text>', '</g></svg>'])
    (OUT/name).write_text('\n'.join(elements)+'\n', encoding='utf-8')

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    diagram('ch01_training_step.svg', 'One optimizer update: token-weighted accumulation',
        [('Count tokens', 'N = all supervised tokens'), ('Micro-steps', 'backward(NLL_sum / N)'), ('Accumulated grad', 'clip once after all steps'), ('Update', 'step, then zero_grad')],
        'Schematic only. No intermediate optimizer update; no project measurements.')
    (OUT/'ch01_lora.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" width="960" height="340" viewBox="0 0 960 340" role="img">\n<title>LoRA parallel paths: frozen base plus trainable low-rank update</title>\n<rect width="960" height="340" fill="#f7f8fc"/>\n<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="#44536d"/></marker></defs>\n<g font-family="sans-serif" fill="#18243b"><text x="35" y="40" font-size="23" font-weight="bold">LoRA: parallel paths, then addition</text>\n<g fill="white" stroke="#637cb2"><rect x="35" y="140" width="120" height="60" rx="10"/><rect x="280" y="65" width="270" height="65" rx="10"/><rect x="280" y="215" width="270" height="65" rx="10"/><circle cx="685" cy="170" r="28"/><rect x="785" y="140" width="140" height="60" rx="10"/></g>\n<g font-size="19"><text x="55" y="177">Input x</text><text x="300" y="103">Frozen base: W x</text><text x="296" y="242">Trainable: (alpha / r) B A x</text><text x="676" y="177">+</text><text x="801" y="177">Output y</text></g>\n<g fill="none" stroke="#44536d" stroke-width="2" marker-end="url(#a)"><path d="M155 170 H205 V97 H278"/><path d="M205 170 V247 H278"/><path d="M550 97 H615 V155 L657 165"/><path d="M550 247 H615 V185 L657 175"/><path d="M713 170 H783"/></g>\n<text x="35" y="320" font-size="16">Schematic only. A: r x d_in; B: d_out x r. Base weights stay frozen.</text></g></svg>\n', encoding='utf-8')
    print('Generated 2 SVG schematics')
