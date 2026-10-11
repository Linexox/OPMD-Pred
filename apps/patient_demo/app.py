"""Patient-first desktop prototype with a separate clinician research view."""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import bootstrap
from core import HISTORY, SITES, SCENARIOS, STATUSES, Store, quality, validate_payload
from export_pdf import export_summary
from inference import ResearchEngine, load_engine_config
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QPixmap, QDesktopServices, QFontDatabase, QFont
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QListWidget, QListWidgetItem, QLineEdit, QTextEdit, QCheckBox, QDateEdit, QFormLayout,
    QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget, QGroupBox, QScrollArea, QFileDialog,
    QMessageBox, QInputDialog, QSplitter, QStackedWidget, QToolButton)

STYLE = '''
QWidget {font-family: "Microsoft YaHei UI"; font-size: 16px; color: #244451;}
QMainWindow {background: #f2f6f7;}
QGroupBox {background: white; border: 1px solid #dce6e8; border-radius: 10px; margin-top: 18px; padding: 16px;}
QGroupBox::title {subcontrol-origin: margin; left: 14px; color: #087f8c; font-weight: bold;}
QLineEdit,QTextEdit,QComboBox,QDateEdit {background: white; padding: 7px; border: 1px solid #cedde1; border-radius: 5px;}
QPushButton {background: #087f8c; color: white; border: 0; padding: 10px 15px; min-height: 24px; border-radius: 6px;}
QPushButton:disabled {background: #aec4c9;} QPushButton:hover {background: #096975;}
QListWidget {background: white; border: 1px solid #dce6e8; border-radius: 8px; padding: 5px;}
QListWidget::item {padding: 9px; border-bottom: 1px solid #eef3f4;}
QListWidget::item:selected {background: #e0f2f2; color: #087f8c;}
QTabWidget::pane {border: 0;} QTabBar::tab {padding: 10px 18px; background: #e4ecee;}
QTabBar::tab:selected {background: white; color: #087f8c;}
QCheckBox {padding: 4px;} QScrollArea {border: 0;}
'''


def text_label(text):
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.PlainText)
    return label


def button(title, callback):
    b = QPushButton(title)
    b.clicked.connect(callback)
    return b


def configure_application(app):
    # The offscreen Qt platform does not enumerate Windows system fonts.
    path = Path('C:/Windows/Fonts/msyh.ttc')
    if path.exists(): QFontDatabase.addApplicationFont(str(path))
    app.setFont(QFont('Microsoft YaHei UI', 10))
    app.setStyleSheet(STYLE)


class InferenceJob(QThread):
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, engine, visit, root):
        super().__init__()
        self.engine, self.visit, self.root = engine, visit, root

    def run(self):
        try:
            self.ready.emit(self.engine.predict(self.visit, self.root))
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}：{exc}')


class Window(QMainWindow):
    def __init__(self, store, engine):
        super().__init__()
        self.store, self.engine = store, engine
        self.visit_id = None
        self.dirty = False
        self._loading = True
        self._active_patient_index = None
        self.image_source, self.report_sources = '', []
        self.job = None
        self.setWindowTitle('口腔黏膜自查与随访助手 · 本地Demo')
        self.resize(1240, 870)
        shell = QWidget()
        layout = QVBoxLayout(shell)
        header = QHBoxLayout()
        title = text_label('口腔黏膜自查与随访助手')
        title.setStyleSheet('font-size:25px; font-weight:bold; color:#183b4b; padding:8px;')
        header.addWidget(title, 1)
        self.role = QComboBox()
        self.role.addItems(['患者视图', '演示医生视图'])
        self.role.currentIndexChanged.connect(self.change_role)
        header.addWidget(self.role)
        layout.addLayout(header)
        layout.addWidget(text_label('准备清楚的资料，接收医生反馈，完成就诊与复诊任务。此版本在电脑上演示流程。'))
        split = QSplitter()
        left = QWidget()
        side = QVBoxLayout(left)
        side.addWidget(text_label('匿名档案'))
        self.patient = QComboBox()
        self.patient.currentIndexChanged.connect(self.patient_changed)
        side.addWidget(self.patient)
        side.addWidget(button('＋ 新建匿名档案', self.add_patient))
        side.addWidget(button('载入两种模拟案例', self.seed_demo))
        side.addWidget(text_label('就诊与提交记录'))
        self.visit_list = QListWidget()
        self.visit_list.setWordWrap(True); self.visit_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.visit_list.currentItemChanged.connect(self.selected_visit)
        side.addWidget(self.visit_list, 1)
        side.addWidget(button('＋ 新建记录', self.new_visit))
        side.addWidget(text_label('同一档案可保留多次记录和不同部位。匿名代号仅用于本地整理。'))
        split.addWidget(left)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.make_next(), '下一步')
        self.tabs.addTab(self.make_entry(), '准备资料')
        self.tabs.addTab(self.make_feedback(), '反馈与历史')
        self.tabs.addTab(self.make_tasks(), '就诊任务')
        self.tabs.addTab(self.make_clinician(), '医生复核与研究')
        self.tabs.setTabVisible(4, False)
        split.addWidget(self.tabs)
        split.setSizes([270, 970])
        layout.addWidget(split, 1)
        self.setCentralWidget(shell)
        self.statusBar().showMessage('离线本地存储；无需注册或上传云端')
        if not self.store.patients():
            self.store.add_patient('演示用户A')
        self.connect_dirty_signals()
        self._loading = False
        self.refresh_patients()

    def make_next(self):
        page = QWidget(); layout = QVBoxLayout(page)
        layout.addWidget(text_label('下一步做什么'))
        layout.addWidget(text_label('查看医生指定的部位和待补资料。提交后，会显示是否仍在等待复核。'))
        self.next_list = QListWidget(); layout.addWidget(self.next_list, 1)
        self.next_list.setWordWrap(True); self.next_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(button('继续选中的任务', lambda: self.continue_task(self.next_list)))
        layout.addWidget(button('准备一次新的就诊资料', self.start_new_entry))
        layout.addWidget(text_label('补资料由医生确认；线下复诊的完成状态由患者自报。'))
        return page

    def start_new_entry(self):
        if self.new_visit(): self.tabs.setCurrentIndex(1)

    def entry_row(self, form, title, field, key):
        box = QWidget(); layout = QVBoxLayout(box); layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field)
        error = text_label(''); error.setStyleSheet('color:#a12632;'); error.hide(); layout.addWidget(error)
        self.error_labels[key] = error; self.input_widgets[key] = field
        form.addRow(title, box)

    def make_entry(self):
        page = QWidget(); outer = QVBoxLayout(page)
        self.step_title = text_label(''); self.step_title.setStyleSheet('font-size:20px;font-weight:bold;padding:8px;')
        outer.addWidget(self.step_title)
        self.task_context = text_label(''); outer.addWidget(self.task_context)
        self.entry_stack = QStackedWidget(); outer.addWidget(self.entry_stack, 1)
        self.error_labels, self.input_widgets, self.fields = {}, {}, {}
        layouts = []
        for _ in range(3):
            body = QWidget(); layout = QVBoxLayout(body); layouts.append(layout)
            scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(body); self.entry_stack.addWidget(scroll)
        one, two, three = layouts
        group = QGroupBox('描述情况'); form = QFormLayout(group)
        self.scenario = QComboBox()
        for k,v in SCENARIOS.items(): self.scenario.addItem(v,k)
        self.scenario.currentIndexChanged.connect(self.update_guide)
        self.site = QComboBox(); self.site.addItems(SITES)
        self.visit_date = QDateEdit(); self.visit_date.setCalendarPopup(True)
        self.visit_date.setDisplayFormat('yyyy-MM-dd'); self.visit_date.setDate(date.today())
        self.entry_row(form,'使用场景',self.scenario,'scenario')
        self.entry_row(form,'记录日期',self.visit_date,'visit_date')
        self.entry_row(form,'病损/拍照部位',self.site,'site')
        self.symptoms = QTextEdit(); self.symptoms.setMinimumHeight(100)
        self.symptoms.setPlaceholderText('何时发现？在哪个部位？有什么自觉变化？')
        self.entry_row(form,'本次情况',self.symptoms,'symptoms')
        self.guide = text_label(''); form.addRow(self.guide)
        one.addWidget(group); one.addStretch()
        photo = QGroupBox('照片与已有资料'); layout = QVBoxLayout(photo)
        layout.addWidget(text_label('镜头清洁、光线均匀、对焦后再拍。按医生指定部位准备照片，保持舒适，不为拍照牵拉疼痛区域。'))
        actions = QHBoxLayout()
        self.photo_button = button('选择照片',self.pick_image); actions.addWidget(self.photo_button)
        self.report_button = button('添加报告',self.pick_reports); actions.addWidget(self.report_button)
        actions.addWidget(button('查看报告',self.open_reports)); layout.addLayout(actions)
        self.image_info, self.report_info = text_label('未选择照片'), text_label('未附报告')
        layout.addWidget(self.image_info); layout.addWidget(self.report_info)
        self.photo_date = QLineEdit(); self.photo_date.setPlaceholderText('实际拍摄日期YYYY-MM-DD；未知留空')
        form = QFormLayout(); self.entry_row(form,'拍摄日期',self.photo_date,'photo_date'); layout.addLayout(form)
        self.preview = QLabel('照片预览'); self.preview.setMinimumHeight(150)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter); self.preview.setStyleSheet('background:#e8f0f1;border-radius:8px;')
        layout.addWidget(self.preview); two.addWidget(photo)
        history = QGroupBox('烟酒槟榔史（未知可留空）'); h = QFormLayout(history)
        for key,title,_ in HISTORY:
            field = QLineEdit(); field.setPlaceholderText('未知留空，真实0填写0')
            self.fields[key] = field; self.entry_row(h,title,field,key)
        two.addWidget(history)
        self.measurement_toggle = QToolButton(); self.measurement_toggle.setText('▸ 检测数值与来源（可由医生协助填写）')
        self.measurement_toggle.setCheckable(True); two.addWidget(self.measurement_toggle)
        self.measurement_body = QGroupBox('核对原报告'); t = QFormLayout(self.measurement_body)
        self.tct = QComboBox(); self.tct.setEditable(True)
        self.tct.addItems(['','未见异常','未见核异质细胞','可疑异型细胞','非典型鳞状上皮细胞','高度可疑异型细胞'])
        self.entry_row(t,'TCT意见',self.tct,'tct')
        for key,title in [('dna_count','DNA异常细胞数'),('dna_total','DNA分析细胞总数'),
                          ('dna_threshold','DNA阈值（2.3或2.5）'),('methylation','甲基化总数')]:
            field=QLineEdit(); field.setPlaceholderText('未知留空，真实0填写0'); self.fields[key]=field
            self.entry_row(t,title,field,key)
        self.source_note=QLineEdit(); self.source_note.setPlaceholderText('注明报告、日期、部位与字段来源')
        self.report_date=QLineEdit(); self.report_date.setPlaceholderText('YYYY-MM-DD；未知留空')
        self.report_site=QComboBox(); self.report_site.addItems(['不确定']+list(SITES))
        for title,field,key in [('来源说明',self.source_note,'source_note'),('报告日期',self.report_date,'report_date'),('取材部位',self.report_site,'report_site')]:
            self.entry_row(t,title,field,key)
        self.confirmed=QCheckBox('已按报告核对日期、部位和检测数值'); t.addRow(self.confirmed)
        for field in [self.fields[k] for k in ('dna_count','dna_total','dna_threshold','methylation')]+[self.source_note,self.report_date]:
            field.textChanged.connect(lambda *_: self.confirmed.setChecked(False))
        self.tct.currentTextChanged.connect(lambda *_: self.confirmed.setChecked(False))
        self.report_site.currentTextChanged.connect(lambda *_: self.confirmed.setChecked(False))
        t.addRow(text_label('没有报告也可准备首次就诊资料。新附件或字段变化后须重新核对；总细胞数0按缺失处理。'))
        two.addWidget(self.measurement_body); self.measurement_body.hide()
        self.measurement_toggle.toggled.connect(self.measurement_body.setVisible)
        two.addStretch()
        three.addWidget(text_label('核对后提交给医生。返回修改不会丢掉本次输入。'))
        self.entry_summary=QTextEdit(); self.entry_summary.setReadOnly(True); three.addWidget(self.entry_summary,1)
        three.addWidget(button('保存并提交复核',self.save_submit))
        navigation=QHBoxLayout()
        self.previous_button=button('上一步',lambda: self.move_step(-1)); navigation.addWidget(self.previous_button)
        navigation.addWidget(button('保存草稿',self.save))
        self.next_button=button('下一步',lambda: self.move_step(1)); navigation.addWidget(self.next_button)
        outer.addLayout(navigation); self.entry_stack.currentChanged.connect(self.refresh_step)
        self.refresh_step(); return page

    def make_feedback(self):
        page = QWidget(); l = QVBoxLayout(page)
        self.feedback = QTextEdit(); self.feedback.setReadOnly(True)
        l.addWidget(self.feedback)
        l.addWidget(text_label('当前记录与此前同部位照片：拍摄角度、光照可能不同。请医生确认对应区域；仅供查看。'))
        row = QHBoxLayout()
        self.before, self.after = QLabel(), QLabel()
        for w in (self.before, self.after):
            w.setAlignment(Qt.AlignmentFlag.AlignCenter); w.setMinimumHeight(200); row.addWidget(w)
        l.addLayout(row)
        row = QHBoxLayout()
        self.before_caption, self.after_caption = text_label(''), text_label('')
        row.addWidget(self.before_caption); row.addWidget(self.after_caption); l.addLayout(row)
        l.addWidget(button('导出就诊摘要PDF', self.export))
        return page

    def make_tasks(self):
        page = QWidget(); l = QVBoxLayout(page)
        l.addWidget(text_label('首次自查：查看提交与医生反馈。已就诊随访：按医生布置的日期和部位完成任务。'))
        self.task_list = QListWidget(); l.addWidget(self.task_list)
        self.task_list.setWordWrap(True); self.task_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        l.addWidget(button('打开任务来源记录', self.open_task_visit))
        l.addWidget(button('继续补充选中任务', lambda: self.continue_task(self.task_list)))
        l.addWidget(button('患者自报已完成线下复诊', self.complete_task))
        l.addWidget(text_label('日期与任务由医生填写；系统只显示待办，不自动制定复诊周期。'))
        return page

    def make_clinician(self):
        page = QWidget(); outer=QVBoxLayout(page)
        self.doctor_tabs=QTabWidget(); outer.addWidget(self.doctor_tabs)
        review_page=QWidget(); l=QVBoxLayout(review_page)
        l.addWidget(text_label('本地医生演示视图：角色切换用于演示，不是医院账号认证。研究输出未经社区初筛和长期癌变风险验证。'))
        form = QFormLayout()
        self.reviewer = QLineEdit(); self.reviewer.setPlaceholderText('复核人匿名代号')
        self.review_text = QTextEdit(); self.review_text.setFixedHeight(100)
        self.needs_more = QCheckBox('需补充资料后再复核')
        self.due = QLineEdit(); self.due.setPlaceholderText('可留空；日期YYYY-MM-DD')
        self.task_note = QLineEdit(); self.task_note.setPlaceholderText('例如：按指定部位准备复诊照片')
        form.addRow('复核人', self.reviewer); form.addRow('患者可见意见', self.review_text)
        self.required_checks = {}
        required = QWidget(); required_layout = QHBoxLayout(required)
        for key,title in [('photo','照片'),('report','检测报告'),('history','病史'),('other','其他')]:
            check = QCheckBox(title); self.required_checks[key] = check; required_layout.addWidget(check)
        self.material_note = QLineEdit(); self.material_note.setPlaceholderText('说清楚要补什么、拍哪个部位或核对哪份报告')
        form.addRow('要求补充资料', required); form.addRow('具体补充说明', self.material_note)
        form.addRow(self.needs_more); form.addRow('医生指定复诊日期', self.due); form.addRow('任务说明', self.task_note)
        l.addLayout(form); l.addWidget(button('确认并发送至患者视图', self.review))
        self.review_error=text_label(''); self.review_error.setStyleSheet('color:#a12632;'); self.review_error.hide(); l.addWidget(self.review_error)
        review_scroll=QScrollArea(); review_scroll.setWidgetResizable(True); review_scroll.setWidget(review_page)
        self.doctor_tabs.addTab(review_scroll,'复核意见与任务')
        research_page=QWidget(); l=QVBoxLayout(research_page)
        l.addWidget(text_label('研究模型：正常/轻度、中度、重度。分数为模型类别概率，未校准为诊断可信度或癌变概率。'))
        self.run_button = button('运行实时研究推理', self.run_inference)
        l.addWidget(self.run_button)
        self.available_card, self.missing_card, self.scores_card = text_label(''), text_label(''), text_label('')
        for title,widget in [('已有资料',self.available_card),('缺失资料',self.missing_card),('三类研究分数',self.scores_card)]:
            group=QGroupBox(title); content=QVBoxLayout(group); content.addWidget(widget); l.addWidget(group)
        self.research = QTextEdit(); self.research.setReadOnly(True); self.research.setMaximumHeight(100); l.addWidget(self.research)
        toggle = QToolButton(); toggle.setText('▸ 查看原始运行记录'); toggle.setCheckable(True); l.addWidget(toggle)
        self.raw_log=QTextEdit(); self.raw_log.setReadOnly(True); self.raw_log.setMinimumHeight(220); self.raw_log.hide()
        toggle.toggled.connect(self.raw_log.setVisible); l.addWidget(self.raw_log)
        research_scroll=QScrollArea(); research_scroll.setWidgetResizable(True); research_scroll.setWidget(research_page)
        self.doctor_tabs.addTab(research_scroll,'研究辅助'); return page

    def connect_dirty_signals(self):
        widgets = list(self.input_widgets.values()) + [self.confirmed]
        for widget in widgets:
            if isinstance(widget,QComboBox): signal=widget.currentTextChanged
            elif isinstance(widget,QDateEdit): signal=widget.dateChanged
            elif isinstance(widget,QCheckBox): signal=widget.toggled
            else: signal=widget.textChanged
            signal.connect(lambda *_values,field=widget: self.mark_dirty(field))

    def mark_dirty(self,field=None):
        if not self._loading:
            self.dirty=True
            for key,widget in self.input_widgets.items():
                if widget is field: self.error_labels[key].hide()

    def payload(self):
        p={k:f.text() for k,f in self.fields.items()}
        p.update(tct=self.tct.currentText(),source_note=self.source_note.text(),report_confirmed=self.confirmed.isChecked(),
                 report_date=self.report_date.text().strip(),report_site=self.report_site.currentText(),photo_date=self.photo_date.text().strip())
        return p

    def validate_entry(self):
        for key,title in [('photo_date','拍摄日期'),('report_date','报告日期')]:
            text=getattr(self,key).text().strip()
            if text:
                try: date.fromisoformat(text)
                except ValueError: raise ValueError(f'{title}请使用YYYY-MM-DD格式，例如2026-10-10')
        return validate_payload(self.payload())

    def refresh_step(self,*_):
        index=self.entry_stack.currentIndex()
        self.step_title.setText(f"第{index+1}步，共3步 · {['描述情况','照片与已有资料','核对提交'][index]}")
        self.previous_button.setEnabled(index>0); self.next_button.setVisible(index<2)
        if index==2:
            p=self.payload(); v=self.store.visit(self.visit_id) if self.visit_id else None
            images=bool(self.image_source or (v and v['image']))
            count=len(self.report_sources)+len(json.loads(v['report'] or '[]') if v else [])
            history=sum(bool(str(p.get(k,'')).strip()) for k,_,_ in HISTORY)
            self.entry_summary.setPlainText(f"场景：{self.scenario.currentText()}\n日期：{self.visit_date.date().toString('yyyy-MM-dd')}\n部位：{self.site.currentText()}\n\n本次情况：\n{self.symptoms.toPlainText() or '尚未填写'}\n\n照片：{'已有' if images else '未提供'}\n报告：{count}份（保存时去重）\n已填写病史字段：{history}项\n\n检测核对：{'已勾选，保存后以来源检查为准' if self.confirmed.isChecked() else '尚待人工核对'}\n\n请确认上述资料对应同一次记录和部位。提交后等待医生复核。")

    def move_step(self,direction):
        if direction>0:
            try: self.validate_entry()
            except ValueError as exc: self.show_validation_error(str(exc)); return
        self.entry_stack.setCurrentIndex(max(0,min(2,self.entry_stack.currentIndex()+direction)))

    def show_validation_error(self,message):
        for error in self.error_labels.values(): error.hide()
        mapping=[('拍摄日期','photo_date'),('报告日期','report_date'),('来源','source_note'),('取材部位','report_site'),
                 ('DNA总','dna_total'),('总数','dna_total'),('阈值','dna_threshold'),('细胞数','dna_count'),
                 ('DNA异常','dna_count'),('甲基化','methylation'),('部位','site')]
        mapping += [(title.split('（')[0].split('(')[0],key) for key,title,_ in HISTORY]
        key=next((key for text,key in mapping if text in message),None)
        if key:
            self.tabs.setCurrentIndex(1); self.entry_stack.setCurrentIndex(0 if key in ('site','visit_date','scenario','symptoms') else 1)
            if key in ('dna_count','dna_total','dna_threshold','methylation','tct','source_note','report_date','report_site'):
                self.measurement_toggle.setChecked(True)
            error=self.error_labels[key]; error.setText('请修改：'+message); error.show()
            self.input_widgets[key].setFocus()
        else:
            self.tabs.setCurrentIndex(1); self.entry_stack.setCurrentIndex(1)
            self.image_info.setText('请核对：'+message)
        self.statusBar().showMessage(message)

    def guard_changes(self):
        if not self.dirty: return True
        dialog=QMessageBox(self); dialog.setWindowTitle('保留本次输入'); dialog.setText('资料尚未保存，接下来怎样处理？')
        save=dialog.addButton('保存草稿',QMessageBox.ButtonRole.AcceptRole)
        discard=dialog.addButton('放弃修改',QMessageBox.ButtonRole.DestructiveRole)
        dialog.addButton('继续编辑',QMessageBox.ButtonRole.RejectRole); dialog.exec()
        if dialog.clickedButton()==save: return bool(self.save())
        if dialog.clickedButton()==discard: self.dirty=False; return True
        return False

    def notify(self, action):
        try: return action()
        except ValueError as exc:
            self.show_validation_error(str(exc)); return None
        except Exception as exc:
            QMessageBox.warning(self, '请核对资料', str(exc))
            return None

    def refresh_patients(self, selected=None):
        self.patient.blockSignals(True); self.patient.clear()
        for p in self.store.patients(): self.patient.addItem(p['alias'], p['id'])
        if selected: self.patient.setCurrentIndex(self.patient.findData(selected))
        self._active_patient_index=None
        self.patient.blockSignals(False); self.patient_changed()

    def add_patient(self):
        if not self.guard_changes(): return
        alias, ok = QInputDialog.getText(self, '匿名档案', '填写演示代号，不填写真实姓名或证件号：')
        if ok and alias.strip():
            pid = self.notify(lambda: self.store.add_patient(alias))
            if pid: self.refresh_patients(pid)

    def seed_demo(self):
        if not self.guard_changes(): return
        from demo_fixtures import seed
        pid = self.notify(lambda: seed(self.store))
        if pid:
            self.refresh_patients(pid)
            if self.visit_list.count(): self.visit_list.setCurrentRow(0)
            self.statusBar().showMessage('已载入虚构案例，模拟反馈不代表医生评审或临床结论。')

    def patient_changed(self, *_):
        desired=self.patient.currentIndex()
        old=self._active_patient_index
        if old is not None and desired != old:
            self.patient.blockSignals(True); self.patient.setCurrentIndex(old); self.patient.blockSignals(False)
            if not self.guard_changes(): return
            self.patient.blockSignals(True); self.patient.setCurrentIndex(desired); self.patient.blockSignals(False)
        self._active_patient_index=desired
        self.refresh_visits(); self.new_visit(force=True); self.refresh_tasks()

    def refresh_visits(self, selected=None):
        self.visit_list.blockSignals(True); self.visit_list.clear()
        pid = self.patient.currentData()
        for v in self.store.visits(pid) if pid else []:
            item = QListWidgetItem(f"{v['visit_date']} · {v['site']}\n{SCENARIOS[v['scenario']]}　{STATUSES[v['status']]}")
            item.setData(Qt.ItemDataRole.UserRole, v['id']); self.visit_list.addItem(item)
            if selected == v['id']: self.visit_list.setCurrentItem(item)
        self.visit_list.blockSignals(False)

    def selected_visit(self, item, *_):
        if not item: return
        desired=item.data(Qt.ItemDataRole.UserRole)
        if desired != self.visit_id and not self.guard_changes():
            self.refresh_visits(self.visit_id); return
        self.visit_id = desired
        self.load_visit()

    def update_guide(self, *_):
        if not hasattr(self, 'guide'): return
        self.guide.setText('首次自查：记录发现的部位和情况，提交给医生复核；有不适请及时就诊。'
                           if self.scenario.currentData() == 'self_check' else
                           '已就诊随访：沿用医生确认的病损部位，填写此次日期，按指定任务补充记录。')

    def new_visit(self,force=False):
        if not force and not self.guard_changes(): return False
        self._loading=True
        for error in self.error_labels.values(): error.hide()
        self.visit_id = None; self.image_source = ''; self.report_sources = []
        for f in self.fields.values(): f.clear()
        self.symptoms.clear(); self.tct.setCurrentText(''); self.source_note.clear(); self.confirmed.setChecked(False)
        self.report_date.clear(); self.report_site.setCurrentIndex(0)
        self.photo_date.clear()
        self.visit_date.setDate(date.today()); self.site.setCurrentIndex(0); self.scenario.setCurrentIndex(0)
        self.update_guide(); self.preview.clear(); self.preview.setText('照片预览')
        self.image_info.setText('尚未选择照片'); self.report_info.setText('尚未附报告')
        self.feedback.setPlainText('新建记录，保存并提交后等待医生复核。')
        self.research.clear(); self.review_text.clear(); self.due.clear(); self.task_note.clear()
        for w in (self.before, self.after): w.clear()
        self.before_caption.clear(); self.after_caption.clear()
        self.site.setEnabled(True); self.scenario.setEnabled(True); self.task_context.clear(); self.entry_stack.setCurrentIndex(0)
        self.material_note.clear()
        for check in self.required_checks.values(): check.setChecked(False)
        self._loading=False; self.dirty=False; self.show_research(None); return True

    def show_image(self, widget, path, size=(500, 190)):
        pix = QPixmap(str(path))
        widget.setPixmap(pix.scaled(*size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def load_visit(self):
        self._loading=True
        for error in self.error_labels.values(): error.hide()
        v = self.store.visit(self.visit_id); p = v['payload']
        self.image_source = ''; self.report_sources = []
        self.scenario.setCurrentIndex(self.scenario.findData(v['scenario'])); self.site.setCurrentText(v['site'])
        self.visit_date.setDate(date.fromisoformat(v['visit_date'])); self.symptoms.setPlainText(v['symptoms'])
        for k, f in self.fields.items(): f.setText('' if p.get(k) is None else str(p[k]))
        self.tct.setCurrentText(p.get('tct', '')); self.source_note.setText(p.get('source_note', ''))
        self.report_date.setText(p.get('report_date', '')); self.report_site.setCurrentText(p.get('report_site', '不确定'))
        self.photo_date.setText(p.get('photo_date', ''))
        self.confirmed.setChecked(p.get('report_confirmed', False))
        if v['image']:
            self.show_image(self.preview, self.store.root / v['image'])
            self.image_info.setText('；'.join(v['quality'].get('messages', [])) or '未触发预设技术提示；病损可见性仍需医生确认。')
        else:
            self.preview.clear(); self.preview.setText('未提供照片'); self.image_info.setText('未提供照片')
        reports = json.loads(v['report'] or '[]')
        self.report_info.setText(f'已附{len(reports)}份报告；' + ('字段已人工核对' if p.get('report_confirmed') else '字段待人工核对'))
        lines = [f"{v['visit_date']}｜{v['site']}｜{STATUSES[v['status']]}", v['symptoms'],
                 '\n医生确认意见：', v['feedback'] or '尚无反馈']
        for past in self.store.feedback_history(v['id']):
            lines.append(f"\n历史反馈（资料版本{past['revision']}，{past['reviewed']}）：{past['feedback']}")
        self.feedback.setPlainText('\n'.join(lines))
        self.review_text.setPlainText(v['feedback']); self.reviewer.setText(v['reviewer'])
        self.needs_more.setChecked(v['status'] == 'needs_more'); self.due.clear(); self.task_note.clear()
        self.material_note.clear()
        for check in self.required_checks.values(): check.setChecked(False)
        self.site.setEnabled(not bool(v['task_id']))
        self.scenario.setEnabled(not bool(v['task_id']))
        if v['task_id']:
            task=self.store.task(v['task_id']); self.task_context.setText('本次补充对应：'+task['site']+' · '+task['note'])
        else: self.task_context.clear()
        self.show_research(self.store.latest_run(v['id']))
        photos = [x for x in self.store.visits(v['patient_id']) if x['site'] == v['site'] and x['image']
                  and (x['visit_date'], x['created'], x['id']) < (v['visit_date'], v['created'], v['id'])]
        photos.sort(key=lambda x: (x['visit_date'], x['created'], x['id']))
        for w in (self.before, self.after): w.clear()
        self.before_caption.clear(); self.after_caption.clear()
        if v['image']:
            self.show_image(self.after, self.store.root / v['image'], (350, 230))
            self.after_caption.setText(f"当前：{v['visit_date']} · {v['site']} · {v['id'][:8]}\n拍摄：{p.get('photo_date') or '日期未知'}")
            if photos and v['site'] != '其他（备注说明）':
                x = photos[-1]
                self.show_image(self.before, self.store.root / x['image'], (350, 230))
                self.before_caption.setText(f"此前：{x['visit_date']} · {x['site']} · {x['id'][:8]}\n拍摄：{x['payload'].get('photo_date') or '日期未知'}")
            else:
                self.before.setText('没有可对应的此前照片')
        else:
            self.after.setText('当前记录没有照片，未生成对照')
        self.refresh_tasks()
        self._loading=False; self.dirty=False; self.refresh_step()

    def pick_image(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择病损照片', '', '图片 (*.jpg *.jpeg *.png)')
        if path:
            q = self.notify(lambda: quality(path))
            if q is not None:
                self.image_source = path; self.show_image(self.preview, path)
                self.dirty=True
                self.image_info.setText('；'.join(q['messages']) or '未触发预设技术提示；病损可见性仍需医生确认。')

    def pick_reports(self):
        paths, _ = QFileDialog.getOpenFileNames(self, '选择检测报告', '', '报告 (*.pdf *.jpg *.jpeg *.png)')
        if paths:
            self.report_sources = list(dict.fromkeys(self.report_sources + paths)); self.confirmed.setChecked(False)
            self.dirty=True
            self.report_info.setText(f'待追加{len(self.report_sources)}份报告，保存后请核对字段并再次勾选确认。')

    def open_reports(self):
        paths = self.report_sources or ([str(self.store.root / p) for p in json.loads(self.store.visit(self.visit_id)['report'] or '[]')] if self.visit_id else [])
        if not paths: self.statusBar().showMessage('尚未附报告'); return
        selected, ok = QInputDialog.getItem(self, '选择附件', '打开本地原报告：', paths, 0, False)
        if ok: QDesktopServices.openUrl(QUrl.fromLocalFile(selected))

    def save(self):
        def action():
            p = self.validate_entry()
            vid = self.store.save_visit(self.patient.currentData(), self.scenario.currentData(), self.site.currentText(),
                 self.visit_date.date().toString('yyyy-MM-dd'), self.symptoms.toPlainText(), p,
                 self.image_source, self.report_sources, self.visit_id)
            self.visit_id = vid; self.refresh_visits(vid); self.load_visit()
            self.dirty=False
            self.statusBar().showMessage('资料已保存。修改后重新提交；先前反馈已保留在历史中。')
            return vid
        return self.notify(action)

    def save_submit(self):
        vid = self.save()
        if vid:
            def action():
                self.store.submit(vid); self.refresh_visits(vid); self.load_visit()
                self.statusBar().showMessage('✓ 已提交，等待医生复核'); return True
            self.notify(action)

    def review(self):
        if not self.visit_id: return
        def action():
            self.store.review(self.visit_id, self.review_text.toPlainText(), self.reviewer.text(),
                              self.needs_more.isChecked(), self.due.text().strip(), self.task_note.text(),
                              [key for key,check in self.required_checks.items() if check.isChecked()], self.material_note.text())
            self.refresh_visits(self.visit_id); self.load_visit(); self.statusBar().showMessage('意见已确认，患者视图可查看')
        try:
            action(); self.review_error.hide()
        except ValueError as exc:
            self.review_error.setText('请核对：'+str(exc)); self.review_error.show()
            (self.material_note if '说明' in str(exc) else self.reviewer if '代号' in str(exc) else self.review_text).setFocus()

    def refresh_tasks(self):
        self.task_list.clear(); self.next_list.clear()
        pid = self.patient.currentData()
        for t in self.store.tasks(pid) if pid else []:
            if t['kind']=='material':
                status={'pending':'○ 待补充','submitted':'◷ 已提交，等待医生','confirmed':'✓ 医生已确认'}.get(t['status'],t['status'])
                title='补资料'
            else:
                status='✓ 患者自报完成' if t['status'] in ('completed','patient_reported') else '○ 待线下复诊'
                title=t['due'] or '日期待明确'
            text=f"{title} · {t['site']} · {status}\n{t['note']}\n来源：{t['source_date']} · {t['visit_id'][:8]} · 版本{t['source_revision']}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, t['id']); self.task_list.addItem(item)
            if t['status'] not in ('confirmed','completed','patient_reported'):
                next_item=QListWidgetItem(text); next_item.setData(Qt.ItemDataRole.UserRole,t['id']); self.next_list.addItem(next_item)
        if self.next_list.count(): self.next_list.setCurrentRow(0)

    def complete_task(self):
        item = self.task_list.currentItem()
        if item:
            tid=item.data(Qt.ItemDataRole.UserRole)
            if self.store.task(tid)['kind']!='appointment':
                self.statusBar().showMessage('补资料需提交并等待医生确认；请点击继续补充'); return
            self.store.complete_task(tid); self.refresh_tasks()

    def continue_task(self,widget):
        item=widget.currentItem()
        if not item: return
        tid=item.data(Qt.ItemDataRole.UserRole)
        if not self.guard_changes(): return
        task=self.store.task(tid)
        if task['kind']=='material' and task['status']=='pending':
            vid=self.notify(lambda: self.store.begin_material_task(tid))
            if vid:
                self.visit_id=vid; self.refresh_visits(vid); self.load_visit(); self.entry_stack.setCurrentIndex(0); self.tabs.setCurrentIndex(1)
        else:
            self.visit_id=task['submission_id'] or task['visit_id']; self.refresh_visits(self.visit_id); self.load_visit(); self.tabs.setCurrentIndex(2)

    def open_task_visit(self):
        item = self.task_list.currentItem()
        if item:
            if not self.guard_changes(): return
            task = next(t for t in self.store.tasks(self.patient.currentData()) if t['id'] == item.data(Qt.ItemDataRole.UserRole))
            self.visit_id = task['visit_id']; self.refresh_visits(self.visit_id); self.load_visit(); self.tabs.setCurrentIndex(2)
            self.statusBar().showMessage(f"任务来自资料版本{task['source_revision']}；该记录如已更新，请结合历史反馈查看。")

    def change_role(self, index):
        self.tabs.setTabVisible(4, index == 1)
        self.tabs.setCurrentIndex(4 if index == 1 else 0)

    def show_research(self, result):
        result=result or {'status':'not_run'}
        names={'image':'图片','history':'病史','tct':'TCT','dna':'DNA','methylation':'甲基化'}
        present=result.get('present',{})
        self.available_card.setText('、'.join(names[k] for k,v in present.items() if v) or '尚无运行记录')
        self.missing_card.setText('、'.join(names[k] for k,v in present.items() if not v) or '尚无运行记录')
        p=result.get('probabilities')
        self.scores_card.setText('　｜　'.join(f'{name}：{value:.1%}' for name,value in zip(('正常/轻度','中度','重度'),p)) if p else '尚未生成研究分数')
        self.scores_card.setMinimumHeight(self.scores_card.fontMetrics().lineSpacing()*2)
        labels={'not_run':'尚未运行','running':'正在后台运行，可继续查看资料','predicted':'运行完成，结果仅供研究参考','abstain':'未生成判断：输入全部缺失或未确认','failed':'运行失败，没有生成替代结果'}
        message=labels.get(result['status'],result['status'])
        if result.get('unknown_tct'): message+='\nTCT描述不在训练词表，请核对原报告。'
        if result.get('simulation'): message='【模拟分数，仅用于界面验收】\n'+message
        self.research.setPlainText(message)
        self.raw_log.setPlainText(json.dumps(result,ensure_ascii=False,indent=2))

    def run_inference(self):
        if self.dirty:
            self.statusBar().showMessage('请先保存当前输入，推理使用已保存的资料版本'); return
        if not self.visit_id:
            self.statusBar().showMessage('请先保存一条记录'); return
        if self.job and self.job.isRunning(): return
        v = self.store.visit(self.visit_id)
        self.run_button.setEnabled(False); self.show_research({'status':'running'})
        job = InferenceJob(self.engine, v, self.store.root); self.job = job
        def record(result):
            self.store.add_run(v['id'], v['revision'], result)
            if self.visit_id == v['id'] and self.store.visit(v['id'])['revision'] == v['revision']: self.show_research(result)
            self.statusBar().showMessage('研究结果已保存，未自动发送给患者')
        job.ready.connect(record)
        job.failed.connect(lambda msg: record({'status': 'failed', 'reason': msg, 'probabilities': None, 'input_revision': v['revision']}))
        job.finished.connect(lambda: self.run_button.setEnabled(True))
        job.start()

    def export(self):
        if not self.visit_id: return
        path, _ = QFileDialog.getSaveFileName(self, '导出就诊摘要', '就诊摘要.pdf', 'PDF (*.pdf)')
        if path:
            if not path.lower().endswith('.pdf'): path += '.pdf'
            result = self.notify(lambda: export_summary(self.store, self.visit_id, path))
            if result: self.statusBar().showMessage('已导出患者摘要，未包含研究模型概率')

    def closeEvent(self, event):
        if self.job and self.job.isRunning():
            self.statusBar().showMessage('推理仍在进行，请结束后关闭。'); event.ignore(); return
        if not self.guard_changes(): event.ignore(); return
        self.store.close(); event.accept()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=bootstrap.storage_root())
    parser.add_argument('--model-config', type=Path, default=bootstrap.APP_DIR / 'models.local.json')
    args = parser.parse_args()
    app = QApplication(sys.argv[:1]); configure_application(app)
    try:
        config = load_engine_config(args.model_config)
    except (FileNotFoundError, KeyError):
        config = {'classifier': '', 'encoder': '', 'device': 'cpu'}
    window = Window(Store(args.data_dir), ResearchEngine(config))
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__': main()
