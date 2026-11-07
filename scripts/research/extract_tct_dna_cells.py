"""Extract embedded cytology fields and DNA nucleus crops without editing originals.

Private outputs retain every source path, PDF SHA, page, image xref and crop box.
Known report layouts are recognized by image geometry and report text; unknown
layouts are listed for review rather than exporting logos or diagnostic tables.
"""
import argparse
import collections
import csv
import hashlib
import html
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pymupdf as fitz


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def write_csv(path, rows, columns):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items() if key in columns})


def scan(roots):
    reports = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix.lower() != '.pdf':
                continue
            with path.open('rb') as stream:
                key = hashlib.file_digest(stream, 'sha256').hexdigest()
            report = reports.setdefault(key, {'sha256': key, 'size': path.stat().st_size, 'sources': []})
            report['sources'].append({'root': str(root), 'path': str(path),
                                      'relative_path': str(path.relative_to(root)), 'patient_folder': path.parent.name})
    return list(reports.values())


def image_candidates(page):
    return [item for item in page.get_image_info(xrefs=True)
            if item['width'] >= 120 and item['height'] >= 100
            and item['bbox'][2] - item['bbox'][0] >= 100
            and item['bbox'][3] - item['bbox'][1] >= 80 and item['xref'] >= 0]


def embedded_image(document, page, item):
    if item['xref'] > 0:
        return document.extract_image(item['xref'])
    for block in page.get_text('dict')['blocks']:
        if block['type'] == 1 and max(abs(a - b) for a, b in zip(block['bbox'], item['bbox'])) < 1:
            return {'image': block['image'], 'ext': block['ext']}
    raise ValueError('Inline cell image could not be located')


def select_images(page):
    images = image_candidates(page)
    # Genowise: two cytology fields and two nucleus mosaics in a 3-column, 2-row panel.
    for first in images:
        box = first['bbox']; width, height = box[2] - box[0], box[3] - box[1]
        group = [im for im in images if abs((im['bbox'][2] - im['bbox'][0]) / width - 1) < .04
                 and abs((im['bbox'][3] - im['bbox'][1]) / height - 1) < .04]
        if len(group) != 6:
            continue
        ordered = sorted(group, key=lambda im: (round(im['bbox'][1] / 5), im['bbox'][0]))
        top, bottom = ordered[:3], ordered[3:]
        if max(im['bbox'][1] for im in top) - min(im['bbox'][1] for im in top) > 3:
            continue
        if max(im['bbox'][1] for im in bottom) - min(im['bbox'][1] for im in bottom) > 3:
            continue
        if not all(abs(a['bbox'][0] - b['bbox'][0]) < 3 for a, b in zip(top, bottom)):
            continue
        if top[0]['bbox'][1] < 100 or bottom[0]['bbox'][1] <= top[0]['bbox'][1] + height * .8:
            continue
        return 'genowise_combined', [(top[0], 'combined/fields', None), (top[1], 'combined/fields', None),
                                     (top[2], 'dna/grids_with_di', (5, 5)),
                                     (bottom[0], 'dna/grids_with_di', (5, 5))]
    text = ''.join(page.get_text().split())
    if '甲基化' in text and not any(word in text for word in ('DNA倍体', 'DNA定量', '非妇科细胞学')):
        return 'methylation', []
    if any(word in text for word in ('细胞DNA定量', '细胞核图像及DI', 'DNA细胞学')):
        right = [im for im in images if im['bbox'][0] > page.rect.width * .45
                 and im['bbox'][3] - im['bbox'][1] > 1.15 * (im['bbox'][2] - im['bbox'][0])]
        if right:
            grid = max(right, key=lambda im: im['bbox'][3] - im['bbox'][1])
            fields = [im for im in images if im['bbox'][0] < page.rect.width * .45
                      and im['bbox'][1] > page.rect.height * .35 and im['bbox'][3] < page.rect.height * .8]
            selected = [(grid, 'dna/grids_with_di', (6, 5))]
            if fields:
                selected.insert(0, (max(fields, key=lambda im: im['width'] * im['height']), 'dna/fields', None))
            return 'kingmed_dna', selected
    if any(word in text for word in ('非妇科细胞学报告', '液基非妇科')):
        fields = [im for im in images if .15 * page.rect.height < im['bbox'][1] < .65 * page.rect.height
                  and im['bbox'][3] < .8 * page.rect.height]
        if fields:
            return 'kingmed_tct', [(max(fields, key=lambda im: im['width'] * im['height']), 'tct/fields', None)]
    if any(word in text for word in ('报告单修改通知', '终止项目', '迟发通知')):
        return 'amendment', []
    return 'unrecognized', []


def nucleus_crops(image, columns, rows, lab):
    array = np.asarray(image.convert('RGB'), dtype=np.int16)
    for row in range(rows):
        for column in range(columns):
            x0, x1 = round(column * image.width / columns), round((column + 1) * image.width / columns)
            y0, y1 = round(row * image.height / rows), round((row + 1) * image.height / rows)
            tile = array[y0:y1, x0:x1]
            red, green, blue = tile[..., 0], tile[..., 1], tile[..., 2]
            if lab == 'kingmed_dna':
                letters = (np.max(tile, axis=2) - np.min(tile, axis=2) > 80)
            else:
                letters = ((red > 120) & (green > 60) & (blue < 120) & (red > blue + 40) & (green > blue + 20)) | ((red > 170) & (green < 110) & (blue < 110) & (red > green + 55))
            positions = np.flatnonzero(letters.sum(axis=1) >= 2)
            positions = positions[positions >= int(tile.shape[0] * .50)]
            stop = int(positions[0]) - 3 if len(positions) else int(tile.shape[0] * .72)
            stop = max(int(tile.shape[0] * .50), min(stop, int(tile.shape[0] * .85)))
            box = (x0 + 1, y0 + 1, max(x0 + 2, x1 - 1), y0 + stop)
            yield row + 1, column + 1, box, image.crop(box)


def main():
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', type=Path, nargs='+', default=[project.parent/'DATASETS/sysu',
                        project.parent/'DATASETS/sysu新数据', project/'data/sysu'])
    parser.add_argument('--output', type=Path, default=project/'data/processed/sysu/tct-dna-cell-images/20261106')
    parser.add_argument('--inventory', type=Path, help='Reuse an already frozen PDF inventory')
    parser.add_argument('--reprocess', nargs='+', help='Reprocess specified PDF SHA prefixes; reuse other existing manifests')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reports = json.loads(args.inventory.read_text(encoding='utf-8')) if args.inventory else scan(args.roots)
    artifacts, report_rows, links, model_links = [], [], [], []
    provenance_path = project/'data/processed/sysu/rebuilt-20261009/record_provenance.private.json'
    provenance = json.loads(provenance_path.read_text(encoding='utf-8')) if provenance_path.exists() else []
    for position, report in enumerate(reports):
        report_id = report['sha256'][:16]
        directory = output/'reports'/report_id
        directory.mkdir(parents=True, exist_ok=True)
        own, page_rows = [], []
        record = {**report, 'report_id': report_id, 'error': None}
        reuse = args.reprocess and not any(report['sha256'].startswith(value) for value in args.reprocess)
        if reuse and (directory/'manifest.private.json').exists():
            record = json.loads((directory/'manifest.private.json').read_text(encoding='utf-8'))
            own, page_rows = record['artifacts'], record['pages']
        else:
            reuse = False
        try:
            with fitz.open(report['sources'][0]['path']) as document:
                for page_number, page in enumerate(document, 1):
                    if reuse:
                        break
                    lab, selected = select_images(page)
                    page_rows.append({'page': page_number, 'layout': lab, 'selected_images': len(selected),
                                      'meaningful_image_candidates': len(image_candidates(page))})
                    for number, (item, kind, grid) in enumerate(selected, 1):
                        extracted = embedded_image(document, page, item)
                        relative = Path('reports')/report_id/kind/f'p{page_number:02d}-{number:02d}.{extracted["ext"]}'
                        path = output/relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(extracted['image'])
                        with Image.open(io.BytesIO(extracted['image'])) as opened:
                            image = opened.convert('RGB')
                        base = {'report_id': report_id, 'pdf_sha256': report['sha256'], 'page': page_number,
                                'xref': item['xref'], 'layout': lab, 'role': kind, 'path': relative.as_posix(),
                                'image_sha256': sha(extracted['image']), 'width': image.width, 'height': image.height,
                                'pdf_bbox': list(item['bbox']), 'original_embedded_bytes': True,
                                'may_contain_lab_selection_boxes': kind.endswith('/fields') and lab != 'kingmed_tct',
                                'contains_di_labels': bool(grid), 'training_ready': False}
                        own.append(base)
                        if grid:
                            for row, column, box, crop in nucleus_crops(image, *grid, lab):
                                crop_rel = Path('reports')/report_id/'dna/nuclei'/f'p{page_number:02d}-{number:02d}-r{row:02d}c{column:02d}.png'
                                (output/crop_rel).parent.mkdir(parents=True, exist_ok=True)
                                crop.save(output/crop_rel)
                                raw = (output/crop_rel).read_bytes()
                                own.append({**base, 'role': 'dna/nuclei', 'path': crop_rel.as_posix(),
                                            'image_sha256': sha(raw), 'width': crop.width, 'height': crop.height,
                                            'parent_image': relative.as_posix(), 'crop_bbox_in_parent': list(box),
                                            'original_embedded_bytes': False, 'contains_di_labels': None,
                                            'di_text_excluded_by': 'known_grid_geometry_and_color_hint_needs_review',
                                            'nucleus_row': row, 'nucleus_column': column,
                                            'low_information_hint': float(np.asarray(crop.convert('L')).std()) < 5,
                                            'has_cell_level_ground_truth': False})
        except Exception as exc:
            record['error'] = type(exc).__name__ + ': ' + str(exc)
        record.update(pages=page_rows, artifacts=own, image_count=len(own),
                      status='extracted' if own else 'error' if record['error'] else 'no_cell_image_identified')
        write_json(directory/'manifest.private.json', record)
        artifacts.extend(own)
        report_rows.append({**{k: record[k] for k in ('report_id', 'sha256', 'size', 'status', 'error', 'image_count')},
                            'page_count': len(page_rows), 'layouts': [p['layout'] for p in page_rows],
                            'source_paths': [s['path'] for s in report['sources']]})
        for source in report['sources']:
            links.append({**source, 'report_id': report_id, 'pdf_sha256': report['sha256'],
                          'status': record['status'], 'image_count': len(own)})
        for row in provenance:
            wrong = set(row.get('wrong_patient_report_shas', []))
            if report['sha256'] in wrong:
                model_links.append({'sample_id': row['sample_id'], 'report_id': report_id,
                                    'binding': 'existing_wrong_patient_exclusion', 'use_for_training': False})
                continue
            dna = row.get('dna') or {}
            exact = dna.get('report_sha256') == report['sha256']
            folders = {str(Path(p)).casefold() for p in row.get('bound_source_folders', [])}
            folder_match = any(str(Path(s['path']).parent).casefold() in folders for s in report['sources'])
            if exact or folder_match:
                model_links.append({'sample_id': row['sample_id'], 'report_id': report_id,
                                    'binding': 'existing_DNA_report_sha256' if exact else 'source_folder_candidate',
                                    'use_for_training': False})
        if position % 25 == 0:
            print(json.dumps({'processed_reports': position + 1, 'total_reports': len(reports),
                              'images': len(artifacts)}), flush=True)
    write_json(output/'inventory.private.json', reports)
    write_json(output/'images.private.json', artifacts)
    write_csv(output/'images.private.csv', artifacts, ['report_id', 'pdf_sha256', 'page', 'xref', 'layout', 'role',
              'path', 'image_sha256', 'width', 'height', 'pdf_bbox', 'parent_image', 'crop_bbox_in_parent',
              'may_contain_lab_selection_boxes', 'contains_di_labels', 'training_ready'])
    write_csv(output/'reports.private.csv', report_rows, ['report_id', 'sha256', 'size', 'status', 'error',
              'image_count', 'page_count', 'layouts', 'source_paths'])
    write_csv(output/'patient_report_links.private.csv', links, ['root', 'path', 'relative_path', 'patient_folder',
              'report_id', 'pdf_sha256', 'status', 'image_count'])
    write_csv(output/'model_record_links.private.csv', model_links,
              ['sample_id', 'report_id', 'binding', 'use_for_training'])
    hashes = collections.defaultdict(list)
    for item in artifacts:
        hashes[item['image_sha256']].append(item['path'])
    write_json(output/'duplicate_images.private.json', {key: value for key, value in hashes.items() if len(value) > 1})
    by_role = dict(collections.Counter(a['role'] for a in artifacts))
    summary = {'source_pdf_paths': len(links), 'unique_pdfs': len(reports),
               'reports_with_images': sum(r['image_count'] > 0 for r in report_rows),
               'reports_without_images': sum(r['image_count'] == 0 for r in report_rows),
               'errors': sum(bool(r['error']) for r in report_rows), 'images_by_role': by_role,
               'all_image_files': len(artifacts), 'unique_image_contents': len(hashes),
               'original_sources_modified': False, 'pymupdf_version': fitz.VersionBind,
               'matched_model_records': len({r['sample_id'] for r in model_links if r['binding'] != 'existing_wrong_patient_exclusion'}),
               'cell_level_labels_assigned': False, 'app_or_training_data_changed': False}
    write_json(output/'summary.json', summary)
    sections = []
    for report in report_rows:
        images = [a for a in artifacts if a['report_id'] == report['report_id'] and a['role'] != 'dna/nuclei']
        if not images:
            continue
        names = sorted({s['patient_folder'] for s in links if s['report_id'] == report['report_id']})
        photos = ''.join(f'<figure><a href="{html.escape(a["path"])}"><img loading="lazy" src="{html.escape(a["path"])}"></a><figcaption>{html.escape(a["role"])} · {a["width"]}×{a["height"]} · 第{a["page"]}页</figcaption></figure>' for a in images)
        sections.append(f'<section><h2>{html.escape(" / ".join(names))}</h2><p>报告 {report["report_id"]} · <a href="reports/{report["report_id"]}/manifest.private.json">原始路径和裁剪记录</a></p><div class="photos">{photos}</div></section>')
    output.joinpath('review.private.html').write_text('<!doctype html><meta charset="utf-8"><title>TCT-DNA报告细胞图像</title><style>body{font:16px system-ui;margin:32px;background:#f5f5f8;color:#222}section{background:white;padding:20px;margin:20px 0;border-radius:12px}.photos{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;width:300px}img{width:100%;height:220px;object-fit:contain;background:#eee}figcaption{font-size:14px}h2{font-size:20px}</style><h1>TCT-DNA报告细胞图像</h1><p>点击图片查看原尺寸。视野图保留报告原有选框；DNA拼图保留DI文字，去文字的网格裁剪在对应报告dna/nuclei目录。原报告未改动，自动裁剪仍需检查，未给单个细胞分配病理标签。</p>'+''.join(sections), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
