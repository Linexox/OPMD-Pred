"""Public Zenodo photos paired with explicitly fictional visit measurements."""
from datetime import date
from pathlib import Path


EXAMPLES = (
    ('photo', '仅照片', '下唇', 'C-10-3-6.jpg', {}),
    ('complete', '照片与检测资料', '舌腹', 'C-10-12-5.jpg',
     {'smoking': 1, 'smoking_years': 10, 'smoking_amount': 5,
      'alcohol': 0, 'alcohol_years': 0, 'alcohol_amount': 0,
      'betel': 0, 'betel_years': 0, 'betel_amount': 0,
      'tct': '未见核异质细胞', 'dna_count': 16, 'dna_total': 1000,
      'dna_threshold': 2.3, 'methylation': 2}),
    ('partial', '部分检测资料', '上唇', 'C-10-2-7.jpg',
     {'smoking': 0, 'alcohol': 0, 'betel': 0,
      'tct': '未见核异质细胞', 'dna_count': 0, 'dna_total': 1000,
      'dna_threshold': 2.5}),
)


def examples(store):
    """Create missing demo visits only; keep edited visits and existing records."""
    from PIL import Image, ImageDraw, ImageFont

    images = Path(__file__).resolve().parents[2] / 'data' / 'zenodo' / 'Images'
    today = date.today().isoformat()
    existing = {p['alias']: p['id'] for p in store.patients()}
    result = []
    for key, title, site, filename, fields in EXAMPLES:
        alias = '模拟：' + title
        pid = existing.get(alias)
        visit = next((v for v in store.visits(pid) if v['payload'].get('demo_key') == key), None) if pid else None
        if visit is None:
            photo = images / filename
            if not photo.is_file():
                continue
            folder = store.root / 'demo-examples' / key
            folder.mkdir(parents=True, exist_ok=True)
            payload = dict(fields, demo_key=key, demo_source='公开Zenodo照片＋虚构就诊数据',
                           photo_date=today, report_confirmed=False)
            report = ''
            if fields:
                payload.update(report_date=today, report_site=site,
                               source_note='虚构检测数据，仅用于演示，不属于照片中的真实患者')
                report_path = folder / '模拟检测资料.png'
                image = Image.new('RGB', (1000, 800), 'white')
                draw = ImageDraw.Draw(image)
                font_path = Path('C:/Windows/Fonts/msyh.ttc')
                font = ImageFont.truetype(str(font_path), 29) if font_path.exists() else ImageFont.load_default()
                lines = ['模拟检测资料', '虚构数值，不对应照片中的真实患者', 'TCT：' + fields['tct'],
                         f"DNA异常细胞数：{fields['dna_count']} / {fields['dna_total']}",
                         f"DI阈值：{fields['dna_threshold']}",
                         '甲基化总数：' + str(fields.get('methylation', '未提供')),
                         '演示日期：' + today]
                for index, line in enumerate(lines):
                    draw.text((45, 45 + index * 85), line, fill='#35313e', font=font)
                image.save(report_path)
                report = str(report_path)
            pid = pid or store.add_patient(alias)
            vid = store.save_visit(pid, 'self_check', site, today,
                                   '【虚构案例】公开口腔照片搭配模拟资料，用于体验分析流程。',
                                   payload, image_source=str(photo), report_source=report)
            if fields:
                confirmed = dict(store.visit(vid)['payload'], report_confirmed=True)
                store.save_visit(pid, 'self_check', site, today,
                                 '【虚构案例】公开口腔照片搭配模拟资料，用于体验分析流程。',
                                 confirmed, visit_id=vid)
            visit = store.visit(vid)
        result.append({'key': key, 'title': title, 'patient_id': pid, 'visit_id': visit['id'],
                       'image': visit['image'], 'note': '公开照片＋虚构就诊数据'})
    return result
