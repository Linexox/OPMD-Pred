"""Export the patient-visible visit summary, excluding research model probabilities."""
import html
from pathlib import Path

import bootstrap  # noqa: F401
from core import SCENARIOS, STATUSES


def export_summary(store, visit_id, output):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    body = ParagraphStyle('body', fontName='STSong-Light', fontSize=11, leading=18, spaceAfter=9)
    title = ParagraphStyle('title', parent=body, fontSize=21, leading=28, textColor=colors.HexColor('#183b4b'), spaceAfter=18)
    sub = ParagraphStyle('sub', parent=body, fontSize=13, leading=20, textColor=colors.HexColor('#087f8c'))
    v = store.visit(visit_id)
    alias = next(p['alias'] for p in store.patients() if p['id'] == v['patient_id'])
    story = []
    def add(text, style=body):
        story.append(Paragraph(html.escape(str(text)).replace('\n', '<br/>'), style))
    add('口腔黏膜自查与随访 · 就诊摘要', title)
    add('演示原型｜本摘要整理用户资料和医生确认意见，未生成自动诊断。')
    add(f"匿名代号：{alias}　记录日期：{v['visit_date']}　版本：{v['revision']}")
    add(f"场景：{SCENARIOS[v['scenario']]}　部位：{v['site']}　状态：{STATUSES[v['status']]}")
    add('本次记录', sub)
    add(v['symptoms'] or '未填写症状或变化记录')
    if v['image']:
        add(f"照片采集日期：{p_date}" if (p_date := v['payload'].get('photo_date')) else '照片采集日期未明确，与记录日期分别保留。')
        q = v['quality']
        add(f"照片：{q['width']} × {q['height']} 像素；技术提示：{'；'.join(q['messages']) or '未触发预设技术提示'}")
        add('技术提示不判断病损是否完整可见，不代表排除疾病。')
    else:
        add('本次未提供照片。')
    p = v['payload']
    add('检测资料', sub)
    add('字段已人工核对' if p.get('report_confirmed') else '字段尚待人工核对')
    add(f"TCT记录：{p.get('tct') or '缺失'}")
    if p.get('dna_count') is not None or p.get('dna_total') is not None:
        add(f"DNA：异常细胞数 {p.get('dna_count')}；分析细胞数 {p.get('dna_total')}；DI阈值 {p.get('dna_threshold')}")
        add(f"异常细胞比例：{p['dna_ratio']:.6%}" if p.get('dna_ratio') is not None else 'DNA比例缺失：不能由总数0或不完整字段计算。')
    else:
        add('DNA：未填写')
    add(f"甲基化总数：{p.get('methylation') if p.get('methylation') is not None else '缺失'}")
    add(f"来源说明：{p.get('source_note') or '见本地所附原报告'}")
    add(f"报告日期：{p.get('report_date') or '未明确'}；取材部位：{p.get('report_site') or '未明确'}")
    add('医生反馈', sub)
    if v['reviewed']:
        add(f"复核代号：{v['reviewer']}　时间：{v['reviewed']}")
        add(v['feedback'])
    else:
        add('尚未完成医生复核；该记录尚不包含医生结论。')
    tasks = [t for t in store.tasks(v['patient_id']) if t['visit_id'] == visit_id]
    if tasks:
        add('医生布置的任务', sub)
        for t in tasks:
            status = ({'pending':'待补充','submitted':'已提交，等待医生','confirmed':'医生已确认'}.get(t['status'],t['status'])
                      if t['kind']=='material' else '患者自报完成' if t['status'] in ('completed','patient_reported') else '待线下复诊')
            add(f"{t['due'] or '资料补充'}｜{t['site']}｜{t['note']}｜{status}")
    story.append(Spacer(1, 12))
    add('研究模型的等级分数不包含在本患者摘要中。复诊安排以医生确认内容为准。')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(output), pagesize=A4, leftMargin=42, rightMargin=42, topMargin=38, bottomMargin=38,
                      title='口腔黏膜自查与随访就诊摘要', author='OPMD学生团队Demo').build(story)
    return Path(output)
