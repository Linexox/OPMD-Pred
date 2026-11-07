"""Group extracted field images by the original sample folder for human selection.

Only the three Chinese grade directories are used. Staining, sampling and report
type are not used to separate candidates or rank them. Native pixels are copied
unchanged; DNA montages and montage-derived nucleus crops are excluded.
"""
import argparse
import collections
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageOps


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def metrics(path):
    with Image.open(path) as opened:
        image = opened.convert('RGB')
    small = ImageOps.contain(image, (384, 384), Image.Resampling.LANCZOS)
    rgb = np.asarray(small, dtype=np.float32)
    gray = np.asarray(small.convert('L'), dtype=np.float32)
    foreground = (rgb.max(2) - rgb.min(2) > 18) & (gray < 235)
    laplacian = gray[1:-1, :-2] + gray[1:-1, 2:] + gray[:-2, 1:-1] + gray[2:, 1:-1] - 4 * gray[1:-1, 1:-1]
    selected = foreground[1:-1, 1:-1]
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    marks = ((red > 215) & (green < 65) & (blue < 65)) | ((green > 210) & (red < 70) & (blue < 70))
    return {'sharpness': float(laplacian[selected].var()) if selected.any() else 0.,
            'foreground_fraction': float(foreground.mean()),
            'contrast': float(gray[foreground].std()) if foreground.any() else 0.,
            'short_side': min(image.size), 'mark_pixel_fraction': float(marks.mean()),
            'width': image.width, 'height': image.height}





def render_gallery(samples, output):
    sections = []
    for sample in samples:
        figures = []
        for item in sample['candidates']:
            picked = item['image_sha256'] == sample['preselected_image']
            url = quote(item['path'], safe='/')
            style = 'picked' if picked else ''
            badge = ' · 预选' if picked else ''
            figures.append(f'<figure class="{style}"><a href="{url}" target="_blank"><img loading="lazy" src="{url}"></a><figcaption>候选{item["candidate_number"]:02d}{badge} · {item["width"]}×{item["height"]}</figcaption></figure>')
        body = ''.join(figures) if figures else '<p class="muted">暂无已提取的细胞视野图</p>'
        sections.append(f'<section><h2>{html.escape(sample["name"])}</h2><p class="muted">{html.escape(sample["group"])}</p><div class="photos">{body}</div></section>')
    count = sum(len(sample['candidates']) for sample in samples)
    ready = sum(bool(sample['candidates']) for sample in samples)
    css = 'body{max-width:1120px;margin:24px auto;padding:0 20px;background:#f5f5f8;color:#252636;font:16px system-ui,"Microsoft YaHei",sans-serif}h1{font-size:25px}h2{font-size:19px;margin:0 0 7px}.muted{color:#777;font-size:14px}section{background:white;padding:22px;margin:20px 0;border-radius:12px}.photos{display:flex;flex-wrap:wrap;gap:16px}figure{margin:0;padding:8px;border:2px solid #eee;border-radius:8px;width:310px}figure.picked{border-color:#8873d5}img{width:100%;height:240px;object-fit:contain;background:#f5f5f5}figcaption{margin-top:8px}'
    page = f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>按文件夹整理的细胞图像</title><style>{css}</style><h1>按文件夹整理的细胞图像</h1><p>等级1/2/3 · {len(samples)}个文件夹 · {ready}个有图 · {count}张候选。紫框为预选，点击图片查看原尺寸。</p>' + ''.join(sections) + '</html>'
    (output/'index.private.html').write_text(page, encoding='utf-8')


def main():
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=project.parent/'DATASETS/sysu/data-20250513')
    parser.add_argument('--extracted', type=Path, default=project/'data/processed/sysu/tct-dna-cell-images/20261106')
    parser.add_argument('--output', type=Path, default=project/'data/processed/sysu/cell-image-samples/20261106')
    parser.add_argument('--overrides', type=Path, help='Optional visual/user preselection overrides')
    args = parser.parse_args()
    source, extracted, output = args.source.resolve(), args.extracted.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    review_path = args.overrides or output/'visual_review_overrides.private.json'
    review = read(review_path) if review_path.is_file() else {}
    overrides = review.get('overrides', {})
    images = [item for item in read(extracted/'images.private.json') if item['role'].endswith('/fields')]
    by_report = collections.defaultdict(list)
    image_metrics = {}
    for item in images:
        by_report[item['report_id']].append(item)
        if item['image_sha256'] not in image_metrics:
            image_metrics[item['image_sha256']] = metrics(extracted/item['path'])
    logs = np.log1p([item['sharpness'] for item in image_metrics.values()])
    low, high = np.percentile(logs, [10, 90])
    for item in image_metrics.values():
        focus = float(np.clip((np.log1p(item['sharpness']) - low) / max(high - low, 1e-8), 0, 1))
        item['rank_score'] = round(100 * (.5 * focus + .2 * min(item['foreground_fraction'] / .30, 1)
            + .2 * min(item['short_side'] / 512, 1) + .1 * min(item['contrast'] / 45, 1)
            - .08 * min(item['mark_pixel_fraction'] / .003, 1)), 2)
    groups = sorted(path for path in source.iterdir() if path.is_dir() and path.name.startswith('等级'))
    folders = {folder: {} for group in groups for folder in sorted(group.iterdir()) if folder.is_dir()}
    report_paths = collections.defaultdict(set)
    for report in read(extracted/'inventory.private.json'):
        report_id = report['sha256'][:16]
        for original in report['sources']:
            path = Path(original['path'])
            if not path.is_relative_to(source):
                continue
            relative = path.relative_to(source)
            if len(relative.parts) < 3:
                continue
            folder = source/relative.parts[0]/relative.parts[1]
            if folder not in folders:
                continue
            report_paths[folder].add(str(path))
            for item in by_report[report_id]:
                entry = folders[folder].setdefault(item['image_sha256'], {'image': item, 'sources': []})
                entry['sources'].append({'pdf': str(path), 'pdf_sha256': report['sha256'],
                                         'page': item['page'], 'xref': item['xref'], 'original_role': item['role']})
    samples, candidate_rows = [], []
    shared = collections.defaultdict(list)
    for folder, candidates in folders.items():
        relative = folder.relative_to(source)
        sample_id = hashlib.sha256(str(relative).encode('utf-8')).hexdigest()[:16]
        target = output/'samples'/relative
        target.mkdir(parents=True, exist_ok=True)
        ordered = sorted(candidates.values(), key=lambda entry: (-image_metrics[entry['image']['image_sha256']]['rank_score'], entry['image']['image_sha256']))
        own = []
        for number, entry in enumerate(ordered, 1):
            original = entry['image']; origin = extracted/original['path']
            path = target/'候选图'/f'{number:02d}{origin.suffix}'
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, path)
            score = image_metrics[original['image_sha256']]
            item = {'sample_id': sample_id, 'candidate_number': number, 'image_sha256': original['image_sha256'],
                    'path': path.relative_to(output).as_posix(), 'width': score['width'], 'height': score['height'],
                    'preselected': number == 1, 'quality_metrics': score, 'sources': entry['sources'],
                    'selection_status': 'automatic_preselection_for_user_review'}
            own.append(item)
            shared[original['image_sha256']].append(str(folder))
        preselected = overrides.get(sample_id, {}).get('image_sha256', own[0]['image_sha256'] if own else None)
        if preselected is not None and preselected not in {item['image_sha256'] for item in own}:
            raise ValueError('Visual selection is not a candidate in this source folder')
        for item in own:
            item['preselected'] = item['image_sha256'] == preselected
            candidate_rows.append({**item, 'source_folder': str(folder)})
        sample = {'sample_id': sample_id, 'name': folder.name, 'group': folder.parent.name,
                  'source_folder': str(folder), 'relative_source_folder': relative.as_posix(),
                  'pdf_paths': sorted(report_paths[folder]), 'candidates': own,
                  'preselected_image': preselected,
                  'preselection_method': 'assistant_visual_review' if sample_id in overrides else 'quality_metric_order',
                  'preselection_note': overrides.get(sample_id, {}).get('note', ''),
                  'grouping_basis': 'original_folder_only', 'stain_or_specimen_used_for_ranking': False}
        write(target/'selection.private.json', sample)
        samples.append(sample)
    write(output/'samples.private.json', samples)
    write(output/'quality_metrics.private.json', image_metrics)
    write(output/'same_images_across_folders.private.json', {key: value for key, value in shared.items() if len(value) > 1})
    with (output/'preselection.private.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sample_id', 'source_folder', 'candidate_number', 'path',
                                                   'image_sha256', 'width', 'height', 'preselected', 'quality_metrics', 'sources'])
        writer.writeheader()
        for row in candidate_rows:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if isinstance(row[key], (list, dict)) else row[key]
                             for key in writer.fieldnames})
    summary = {'source_root': str(source), 'source_groups': [g.name for g in groups],
               'sample_folders': len(samples), 'folders_with_candidates': sum(bool(s['candidates']) for s in samples),
               'folders_without_candidates': sum(not s['candidates'] for s in samples),
               'candidate_images': len(candidate_rows), 'preselected_images': sum(s['preselected_image'] is not None for s in samples),
               'visual_review_adjustments': len(overrides),
               'excluded_roles': ['dna/grids_with_di', 'dna/nuclei'],
               'original_images_changed': False, 'clinical_quality_review_performed': False,
               'same_image_across_multiple_folders': sum(len(v) > 1 for v in shared.values()),
               'rank_rule': '50% foreground sharpness, 20% visible-region coverage, 20% resolution, 10% contrast, up to 8% mark-color penalty; engineering heuristic, not clinical grade',
               'stain_and_specimen_ignored': True}
    write(output/'summary.json', summary)
    render_gallery(samples, output)
    old_entry = extracted/'review.private.html'
    redirect = quote(os.path.relpath(output/'index.private.html', old_entry.parent).replace('\\', '/'), safe='/')
    old_entry.write_text(f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url={redirect}"><title>按文件夹挑选细胞图</title><p><a href="{redirect}">打开按原始文件夹整理的细胞图像</a></p>', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
