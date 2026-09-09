#!/usr/bin/env python3
"""Mac .app GUI for Highway Reporter."""
import json
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import default_output_dir
from src.preflight import load_api_key, load_dotenv
from src.report_platforms import (
    DEFAULT_REGION_ID,
    city_by_id,
    cities_in_province,
    copy_all,
    load_preferred_region_id,
    locate_from_ip,
    provinces,
    reward_regions,
    save_preferred_region_id,
)
from src.wechat_pack import (
    ensure_pack_server,
    lan_urls,
    open_wechat,
    reveal_path,
    write_wechat_pack,
)

BG = '#F4EFE4'
SURFACE = '#FFFBF5'
INK = '#1C1914'
MUTED = '#6B6458'
LINE = '#D9D0C0'
AMBER = '#C9A227'
AMBER_HOVER = '#D4B44A'
RED = '#C4452D'
GREEN = '#2F6F4E'
ASPHALT = '#2A2722'


def _fmt_size(path: Path) -> str:
    if not path.is_file():
        return ''
    try:
        n = path.stat().st_size
    except OSError:
        return ''
    if n < 1024 * 1024:
        return f'{n / 1024:.0f} KB'
    return f'{n / (1024 * 1024):.1f} MB'


def _ghost(parent, text, command):
    return tk.Button(
        parent, text=text, command=command,
        bg=SURFACE, fg=INK, activebackground=LINE, activeforeground=INK,
        font=('PingFang SC', 13), relief='flat', padx=14, pady=8,
        highlightthickness=0, cursor='hand2',
    )


def _primary(parent, text, command):
    return tk.Button(
        parent, text=text, command=command,
        bg=AMBER, fg=INK, activebackground=AMBER_HOVER, activeforeground=INK,
        font=('PingFang SC', 16, 'bold'), relief='flat', padx=22, pady=12,
        highlightthickness=0, cursor='hand2',
    )


class ProgressBar(tk.Canvas):
    def __init__(self, master, **kw):
        super().__init__(master, height=12, bg=LINE, highlightthickness=0, **kw)
        self._pct = 0
        self.bind('<Configure>', lambda _e: self.draw())

    def set_pct(self, pct):
        self._pct = max(0, min(100, pct))
        self.draw()

    def draw(self):
        self.delete('all')
        w = max(self.winfo_width(), 1)
        self.create_rectangle(0, 0, int(w * self._pct / 100), 12, fill=AMBER, width=0)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('占用应急车道举报')
        self.geometry('860x760')
        self.minsize(760, 660)
        self.configure(bg=BG)
        self.log_q = queue.Queue()
        self.proc = None
        self._busy = False
        self._city_id = load_preferred_region_id() or DEFAULT_REGION_ID
        self._report = None
        self._out_dir = None
        self._photos = []
        self._reward_only = tk.BooleanVar(value=True)
        self._province_var = tk.StringVar()
        self._city_var = tk.StringVar()

        load_dotenv()

        self.outer = tk.Frame(self, bg=BG, padx=28, pady=22)
        self.outer.pack(fill='both', expand=True)

        head = tk.Frame(self.outer, bg=BG)
        head.pack(fill='x')
        tk.Label(head, text='占用应急车道举报', bg=BG, fg=INK, font=('Songti SC', 24)).pack(side='left')
        self.status = tk.Label(
            head, text='可以开始', bg=ASPHALT, fg=SURFACE,
            font=('PingFang SC', 13), padx=12, pady=4,
        )
        self.status.pack(side='right', pady=4)
        tk.Label(
            self.outer, text='一共三步：选视频  →  等电脑看完  →  发到手机微信去交',
            bg=BG, fg=MUTED, font=('PingFang SC', 13),
        ).pack(anchor='w', pady=(6, 18))

        self.work = tk.Frame(self.outer, bg=BG)
        self.work.pack(fill='both', expand=True)
        self._build_work()

        self.results = tk.Frame(self.outer, bg=BG)

        self.after(120, self.drain_log)
        self.protocol('WM_DELETE_WINDOW', self.on_close)

    def _build_work(self):
        self.video_var = tk.StringVar()
        self.out_var = tk.StringVar(value=str(default_output_dir()))

        tk.Label(
            self.work, text='第 1 步  选出记录仪视频',
            bg=BG, fg=INK, font=('PingFang SC', 15, 'bold'),
        ).pack(anchor='w')
        self.drop = tk.Frame(self.work, bg=SURFACE, highlightthickness=1, highlightbackground=LINE)
        self.drop.pack(fill='x', pady=(8, 0))
        inner = tk.Frame(self.drop, bg=SURFACE, padx=20, pady=18)
        inner.pack(fill='x')
        self.drop_title = tk.Label(
            inner, text='还没有选视频', bg=SURFACE, fg=INK,
            font=('PingFang SC', 16, 'bold'),
        )
        self.drop_title.pack(anchor='w')
        self.drop_sub = tk.Label(
            inner, text='把记录仪插上电脑后，点下面按钮把视频选进来。',
            bg=SURFACE, fg=MUTED, font=('PingFang SC', 13),
        )
        self.drop_sub.pack(anchor='w', pady=(4, 12))
        pick_row = tk.Frame(inner, bg=SURFACE)
        pick_row.pack(anchor='w')
        _primary(pick_row, '选出视频', self.pick_video).pack(side='left')
        _ghost(pick_row, '选出整个文件夹', self.pick_folder).pack(side='left', padx=(10, 0))
        for w in (self.drop, inner, self.drop_title, self.drop_sub):
            w.bind('<Button-1>', lambda _e: self.pick_video())

        tk.Label(
            self.work, text='第 2 步  让电脑帮你找占用应急车道的车',
            bg=BG, fg=INK, font=('PingFang SC', 15, 'bold'),
        ).pack(anchor='w', pady=(22, 8))
        btns = tk.Frame(self.work, bg=BG)
        btns.pack(fill='x')
        self.run_btn = _primary(btns, '开始查找', self.toggle_run)
        self.run_btn.pack(side='left')
        self.last_btn = _ghost(btns, '看上次找到的', self.show_last_result)
        self.last_btn.pack(side='left', padx=(10, 0))
        self._refresh_last_btn()

        self.progress_card = tk.Frame(
            self.work, bg=SURFACE, highlightthickness=1, highlightbackground=LINE,
        )
        self.progress_card.pack(fill='x', pady=(18, 0))
        inner_p = tk.Frame(self.progress_card, bg=SURFACE, padx=20, pady=18)
        inner_p.pack(fill='x')
        self.phase = tk.Label(
            inner_p, text='选好视频后，点「开始查找」', bg=SURFACE, fg=INK,
            font=('PingFang SC', 16, 'bold'),
        )
        self.phase.pack(anchor='w')
        self.phase_sub = tk.Label(
            inner_p, text='查找时请不要关这个窗口，也不要拔记录仪。',
            bg=SURFACE, fg=MUTED, font=('PingFang SC', 13),
        )
        self.phase_sub.pack(anchor='w', pady=(4, 12))
        self.bar = ProgressBar(inner_p)
        self.bar.pack(fill='x')

        self.video_var.trace_add('write', lambda *_: self._refresh_video())

    def _refresh_last_btn(self):
        out = Path(self.out_var.get().strip() or default_output_dir())
        has = (out / 'report.json').exists()
        self.last_btn.configure(state='normal' if has else 'disabled')

    def _set_status(self, text, color=None):
        self.status.configure(text=text, fg=color or SURFACE)

    def _set_phase(self, title, sub, pct):
        self.phase.configure(text=title)
        self.phase_sub.configure(text=sub)
        self.bar.set_pct(pct)
        self._set_status('请稍等', AMBER)

    def _refresh_video(self):
        path = Path(self.video_var.get().strip())
        if not path.exists():
            self.drop_title.configure(text='还没有选视频')
            self.drop_sub.configure(text='把记录仪插上电脑后，点下面按钮把视频选进来。')
            self.drop.configure(highlightbackground=LINE)
            return
        size = _fmt_size(path)
        extra = f'（{size}）' if size else ''
        self.drop_title.configure(text=f'已选好：{path.name}')
        if path.is_dir():
            self.drop_sub.configure(text=f'这是一个文件夹{extra}。下一步：点「开始查找」。')
        else:
            self.drop_sub.configure(text=f'视频已选好{extra}。下一步：点「开始查找」。')
        self.drop.configure(highlightbackground=AMBER)
        self.phase.configure(text='视频已选好，可以开始查找')
        self.phase_sub.configure(text='点黄色的「开始查找」。查找时请不要关窗口。')

    def pick_video(self):
        path = filedialog.askopenfilename(
            title='选出记录仪里的视频',
            filetypes=[('视频', '*.mp4 *.mov *.avi *.mkv *.ts'), ('全部', '*')],
        )
        if path:
            self.video_var.set(path)

    def pick_folder(self):
        path = filedialog.askdirectory(title='选出装着视频的文件夹')
        if path:
            self.video_var.set(path)

    def toggle_run(self):
        if self._busy:
            self.stop()
        else:
            self.start()

    def start(self):
        if self.proc:
            return
        video = self.video_var.get().strip()
        if not video or not Path(video).exists():
            messagebox.showwarning('还没选视频', '请先点「选出视频」，把记录仪里的视频选进来。')
            return
        try:
            load_api_key()
        except RuntimeError:
            messagebox.showerror(
                '现在还不能用',
                '这个软件还没准备好。\n请联系发给你这个软件的人。',
            )
            return

        self._show_work()
        out = self.out_var.get().strip() or str(default_output_dir())
        Path(out).mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, '-u', str(ROOT / 'full_pipeline.py'), video, '-o', out]
        self._busy = True
        self.run_btn.configure(text='停止查找', bg=SURFACE, fg=INK, activebackground=LINE)
        self._set_phase('正在打开视频', '请稍等，不要关这个窗口。', 4)
        threading.Thread(target=self._run, args=(cmd, out), daemon=True).start()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass
            self._set_phase('已经停下', '可以重新点「开始查找」。', self.bar._pct)
            self._set_status('已停下', MUTED)

    def _run(self, cmd, out):
        env = os.environ.copy()
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, env=env,
            )
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self.log_q.put(('LINE', line))
            code = self.proc.wait()
            if code == 0:
                self.log_q.put(('OK', out))
            else:
                self.log_q.put(('FAIL', code))
        except Exception as exc:
            self.log_q.put(('FAIL', str(exc)))
        finally:
            self.proc = None
            self.log_q.put(('DONE', None))

    def _on_line(self, line):
        text = line.strip()
        if '[L1]' in text:
            self._set_phase('正在看视频', '正在找有没有车开在应急车道上。', 16)
        elif '[L1.5]' in text:
            self._set_phase('正在截照片', '挑几张比较清楚的画面。', 30)
        elif '[L2]' in text:
            self._set_phase('正在看车牌', '这一步比较慢，可能要等几分钟。', 42)
        elif '读车牌' in text:
            found = re.search(r'(\d+)/(\d+)', text)
            if found:
                i, n = int(found.group(1)), max(int(found.group(2)), 1)
                pct = 42 + int(40 * i / n)
                self._set_phase('正在看车牌', f'还在看，请继续等。（{i}/{n}）', pct)
        elif '读时间' in text:
            self._set_phase('正在看车牌', '再看一下时间和车的颜色。', 84)
        elif '[L3]' in text:
            self._set_phase('正在准备材料', '马上就可以发到手机了。', 92)

    def drain_log(self):
        while True:
            try:
                item = self.log_q.get_nowait()
            except queue.Empty:
                break
            kind = item[0] if isinstance(item, tuple) else None
            if kind == 'LINE':
                self._on_line(item[1])
            elif kind == 'DONE':
                self._busy = False
                self.run_btn.configure(
                    text='开始查找', bg=AMBER, fg=INK, activebackground=AMBER_HOVER,
                )
                self._refresh_last_btn()
            elif kind == 'OK':
                self._set_phase('找完了', '下一步：发到手机微信去交。', 100)
                self._set_status('找到了', GREEN)
                self.show_results(item[1])
            elif kind == 'FAIL':
                self._set_phase('没看成', '请换一段视频，再点「开始查找」。', self.bar._pct)
                self._set_status('没看成', RED)
        self.after(120, self.drain_log)

    def show_last_result(self):
        out = Path(self.out_var.get().strip() or default_output_dir())
        if not (out / 'report.json').exists():
            messagebox.showinfo('还没有找到过', '请先选出视频，再点「开始查找」。')
            return
        self.show_results(str(out))

    def _show_work(self):
        self.results.pack_forget()
        self.work.pack(fill='both', expand=True)
        self._refresh_last_btn()

    def show_results(self, out_dir):
        path = Path(out_dir) / 'report.json'
        try:
            self._report = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            messagebox.showerror('打不开', '上次的结果打不开，请重新查找一次。')
            return
        self._out_dir = Path(out_dir)
        self.work.pack_forget()
        for child in self.results.winfo_children():
            child.destroy()
        self.results.pack(fill='both', expand=True)
        self._build_results()

    def _build_results(self):
        top = tk.Frame(self.results, bg=BG)
        top.pack(fill='x')
        _ghost(top, '回上一页', self._show_work).pack(side='left')
        vs = self._report.get('violations') or []
        tk.Label(
            top, text=f'找到 {len(vs)} 辆车',
            bg=BG, fg=MUTED, font=('PingFang SC', 13),
        ).pack(side='right')

        tk.Label(
            self.results, text='第 3 步  选好在哪个城市交，然后发到手机',
            bg=BG, fg=INK, font=('PingFang SC', 15, 'bold'),
        ).pack(anchor='w', pady=(16, 8))

        picker = tk.Frame(self.results, bg=SURFACE, highlightthickness=1, highlightbackground=LINE)
        picker.pack(fill='x')
        inner_p = tk.Frame(picker, bg=SURFACE, padx=16, pady=14)
        inner_p.pack(fill='x')
        tk.Label(
            inner_p, text='这段视频是在哪个城市拍的？',
            bg=SURFACE, fg=INK, font=('PingFang SC', 14, 'bold'),
        ).pack(anchor='w')
        row = tk.Frame(inner_p, bg=SURFACE)
        row.pack(fill='x', pady=(10, 8))
        tk.Label(row, text='省', bg=SURFACE, fg=MUTED, font=('PingFang SC', 13)).pack(side='left')
        self.prov_cb = ttk.Combobox(
            row, textvariable=self._province_var, state='readonly',
            width=10, font=('PingFang SC', 13),
        )
        self.prov_cb.pack(side='left', padx=(6, 14))
        tk.Label(row, text='市', bg=SURFACE, fg=MUTED, font=('PingFang SC', 13)).pack(side='left')
        self.city_cb = ttk.Combobox(
            row, textvariable=self._city_var, state='readonly',
            width=14, font=('PingFang SC', 13),
        )
        self.city_cb.pack(side='left', padx=(6, 12))
        self.locate_btn = _ghost(row, '帮我选当地', self._locate_ip)
        self.locate_btn.pack(side='left')
        tk.Checkbutton(
            inner_p, text='只看好给钱的地方', variable=self._reward_only,
            command=self._on_reward_filter,
            bg=SURFACE, fg=INK, activebackground=SURFACE, activeforeground=INK,
            font=('PingFang SC', 13), highlightthickness=0, selectcolor=SURFACE,
        ).pack(anchor='w')
        self.city_hint = tk.Label(
            inner_p, text='', bg=SURFACE, fg=MUTED, font=('PingFang SC', 13),
            wraplength=760, justify='left',
        )
        self.city_hint.pack(anchor='w', pady=(8, 0))

        send_wrap = tk.Frame(self.results, bg=BG)
        send_wrap.pack(fill='x', pady=(14, 8))
        self.send_all_btn = _primary(send_wrap, '把找到的车发到手机微信', self._send_all_wechat)
        self.send_all_btn.pack(fill='x')
        tk.Label(
            self.results,
            text='点上面这个黄按钮。电脑微信打开后，把文件拖给「文件传输助手」。',
            bg=BG, fg=MUTED, font=('PingFang SC', 13),
        ).pack(anchor='w', pady=(0, 10))

        canvas = tk.Canvas(self.results, bg=BG, highlightthickness=0)
        scroll = tk.Scrollbar(self.results, command=canvas.yview)
        self.cards = tk.Frame(canvas, bg=BG)
        self.cards.bind(
            '<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all')),
        )
        self._cards_win = canvas.create_window((0, 0), window=self.cards, anchor='nw')
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')
        canvas.bind('<Configure>', lambda e: canvas.itemconfigure(self._cards_win, width=e.width))

        self.prov_cb.bind('<<ComboboxSelected>>', self._on_province)
        self.city_cb.bind('<<ComboboxSelected>>', self._on_city)
        self._refresh_place_menus(select=True, persist=False)

    def _place_options(self):
        reward_only = self._reward_only.get()
        region = city_by_id(self._city_id)
        if reward_only and not region.get('reward'):
            same = cities_in_province(region['province'], reward_only=True)
            if same:
                region = same[0]
            elif reward_regions():
                region = reward_regions()[0]
            self._city_id = region['id']
        provs = provinces(reward_only=reward_only) or provinces(reward_only=False)
        cities = cities_in_province(region['province'], reward_only=reward_only)
        if not cities:
            cities = cities_in_province(region['province'], reward_only=False)
        return provs, cities, region

    def _refresh_place_menus(self, select=False, persist=False):
        provs, cities, region = self._place_options()
        self.prov_cb['values'] = provs
        labels = []
        ids = []
        for c in cities:
            label = c['city'] + ('  ·给钱' if c.get('reward') else '')
            labels.append(label)
            ids.append(c['id'])
        self.city_cb['values'] = labels
        self._city_labels = dict(zip(labels, ids))
        if select:
            self._province_var.set(region['province'])
            current = next((l for l, i in self._city_labels.items() if i == region['id']), labels[0] if labels else '')
            self._city_var.set(current)
            self._select_city(region['id'], persist=persist)

    def _on_reward_filter(self):
        self._refresh_place_menus(select=True, persist=True)

    def _on_province(self, _event=None):
        province = self._province_var.get()
        cities = cities_in_province(province, reward_only=self._reward_only.get())
        if not cities:
            cities = cities_in_province(province, reward_only=False)
        if not cities:
            return
        current = city_by_id(self._city_id)
        pick = current['id'] if current['province'] == province else cities[0]['id']
        if pick not in {c['id'] for c in cities}:
            pick = cities[0]['id']
        self._city_id = pick
        self._refresh_place_menus(select=True, persist=True)

    def _on_city(self, _event=None):
        region_id = self._city_labels.get(self._city_var.get())
        if region_id:
            self._select_city(region_id)

    def _locate_ip(self):
        self.locate_btn.configure(state='disabled', text='正在选…')

        def work():
            region, status = locate_from_ip()
            self.after(0, lambda r=region, s=status: self._apply_ip(r, s))

        threading.Thread(target=work, daemon=True).start()

    def _apply_ip(self, region, status):
        self.locate_btn.configure(state='normal', text='帮我选当地')
        if not region:
            self.city_hint.configure(text='没能自动选到。请自己点上面的省、市。')
            return
        if not region.get('reward'):
            self._reward_only.set(False)
        self._city_id = region['id']
        self._refresh_place_menus(select=True, persist=True)

    def _how_to_submit(self, city):
        money = '交了可能给钱。' if city.get('reward') else '这里交了不一定给钱。'
        where = city.get('channel') or '打开当地交警的微信'
        return f'{money}交的时候去：{where}'

    def _select_city(self, city_id, persist=True):
        self._city_id = city_id
        city = city_by_id(city_id)
        self.city_hint.configure(text=self._how_to_submit(city))
        if persist:
            save_preferred_region_id(city_id)
        for child in self.cards.winfo_children():
            child.destroy()
        self._photos = []
        vs = self._report.get('violations') or []
        if not vs:
            tk.Label(
                self.cards,
                text='这段视频里，没有找到占用应急车道的车。\n请回上一页，换一段视频再找。',
                bg=BG, fg=MUTED, font=('PingFang SC', 15), justify='left',
            ).pack(anchor='w', pady=20)
            self.send_all_btn.configure(state='disabled')
            return
        self.send_all_btn.configure(state='normal')
        for i, v in enumerate(vs, 1):
            self._violation_card(v, city, i, len(vs))

    def _violation_card(self, v, city, index, total):
        card = tk.Frame(self.cards, bg=SURFACE, highlightthickness=1, highlightbackground=LINE)
        card.pack(fill='x', pady=(0, 12))
        inner = tk.Frame(card, bg=SURFACE, padx=16, pady=14)
        inner.pack(fill='x')
        head = tk.Frame(inner, bg=SURFACE)
        head.pack(fill='x')
        plate = v.get('plate') or '车牌没看清'
        tk.Label(
            head, text=f'第 {index} 辆  ·  {plate}',
            bg=SURFACE, fg=INK, font=('PingFang SC', 16, 'bold'),
        ).pack(side='left')
        ok = bool(v.get('can_report'))
        tk.Label(
            head, text='可以交了' if ok else '还要自己补几项',
            bg=SURFACE, fg=GREEN if ok else RED, font=('PingFang SC', 13),
        ).pack(side='right')

        summary = tk.Label(
            inner,
            text=self._card_summary(v, ok),
            bg=SURFACE, fg=MUTED, font=('PingFang SC', 13),
            justify='left', wraplength=700, anchor='w',
        )
        summary.pack(anchor='w', pady=(8, 10))

        _primary(inner, '把这辆发到手机微信', lambda v=v: self._send_wechat([v])).pack(anchor='w')

        copy_box = tk.Frame(inner, bg=SURFACE)
        rows = city['fields'](v)
        for label, value in rows:
            row = tk.Frame(copy_box, bg=SURFACE)
            row.pack(fill='x', pady=4)
            tk.Label(
                row, text=label, bg=SURFACE, fg=MUTED, font=('PingFang SC', 13),
                width=12, anchor='w',
            ).pack(side='left')
            tk.Label(
                row, text=value or '还没有，请自己补', bg=SURFACE, fg=INK,
                font=('PingFang SC', 14), wraplength=460, justify='left', anchor='w',
            ).pack(side='left', fill='x', expand=True)
            tk.Button(
                row, text='抄下来', command=lambda t=value: self._copy(t),
                bg=SURFACE, fg=INK, activebackground=LINE,
                font=('PingFang SC', 13), relief='flat', padx=10, pady=4,
                highlightthickness=1, highlightbackground=LINE, cursor='hand2',
            ).pack(side='right')
        tk.Button(
            copy_box, text='把这辆的字全部抄下来',
            command=lambda r=rows: self._copy(copy_all(r)),
            bg=SURFACE, fg=INK, font=('PingFang SC', 13), relief='flat',
            padx=12, pady=6, highlightthickness=1, highlightbackground=LINE, cursor='hand2',
        ).pack(anchor='w', pady=(6, 0))

        toggle_btn = _ghost(inner, '不发微信，就在电脑上抄', lambda: None)
        toggle_btn.pack(anchor='w', pady=(8, 0))

        def toggle():
            if copy_box.winfo_ismapped():
                copy_box.pack_forget()
                toggle_btn.configure(text='不发微信，就在电脑上抄')
            else:
                copy_box.pack(fill='x', pady=(10, 0))
                toggle_btn.configure(text='收起抄写')

        toggle_btn.configure(command=toggle)

    def _card_summary(self, v, ok):
        time = v.get('time') or '时间还不知道'
        loc = v.get('location') or '地点还不知道，交的时候看着路牌补上'
        if ok:
            extra = '材料比较齐，可以交。'
        else:
            extra = '有的格子是空的，交的时候自己补上就行。'
        return f'时间：{time}\n地点：{loc}\n{extra}'

    def _send_all_wechat(self):
        vs = self._report.get('violations') or []
        if not vs:
            messagebox.showinfo('没有车', '这段视频里没有找到占用应急车道的车。')
            return
        self._send_wechat(vs)

    def _send_wechat(self, violations):
        if not self._out_dir:
            return
        city = city_by_id(self._city_id)
        self._set_status('请稍等', AMBER)
        if hasattr(self, 'send_all_btn'):
            self.send_all_btn.configure(state='disabled', text='正在准备…')

        def work():
            pack = err = None
            try:
                pack = write_wechat_pack(self._out_dir, violations, city)
            except Exception as exc:
                err = exc
            self.after(0, lambda p=pack, e=err: self._after_wechat_pack(p, e))

        threading.Thread(target=work, daemon=True).start()

    def _after_wechat_pack(self, pack, err):
        if hasattr(self, 'send_all_btn'):
            self.send_all_btn.configure(state='normal', text='把找到的车发到手机微信')
        if err or pack is None:
            messagebox.showerror('没准备好', '材料没准备好，请再点一次黄按钮。')
            self._set_status('找到了', GREEN)
            return
        target = pack.html_files[0] if len(pack.html_files) == 1 else pack.folder
        reveal_path(target)
        opened = open_wechat()
        urls = []
        try:
            port = ensure_pack_server(pack.folder)
            if pack.html_files:
                urls = lan_urls(pack.html_files[0].name, port)
        except OSError:
            pass
        self._show_wechat_guide(pack, opened, urls)
        self._set_status('去发微信', GREEN)

    def _show_wechat_guide(self, pack, opened, urls):
        win = tk.Toplevel(self)
        win.title('发到手机')
        win.configure(bg=BG)
        win.geometry('580x620')
        inner = tk.Frame(win, bg=BG, padx=22, pady=18)
        inner.pack(fill='both', expand=True)
        tk.Label(
            inner, text='按这 4 步发给手机',
            bg=BG, fg=INK, font=('Songti SC', 22),
        ).pack(anchor='w')
        ready = '电脑微信已经打开。' if opened else '请先点下面「打开电脑微信」。'
        tk.Label(
            inner, text=ready,
            bg=BG, fg=MUTED, font=('PingFang SC', 13),
        ).pack(anchor='w', pady=(4, 12))

        n = len(pack.html_files)
        file_word = '那一个文件' if n == 1 else f'那 {n} 个文件，一次拖一个'
        steps = (
            '1. 点下面黄色按钮，打开电脑微信\n\n'
            '2. 在微信里点「文件传输助手」\n'
            '    （就是自己跟自己聊天那个）\n\n'
            f'3. 把已经弹出的 {file_word}，拖进对话框，点发送\n\n'
            '4. 拿出手机，打开微信，点文件传输助手，\n'
            '    点开刚发来的文件，按上面的提示去交'
        )
        tk.Label(
            inner, text=steps, bg=SURFACE, fg=INK, font=('PingFang SC', 15),
            justify='left', wraplength=520, padx=16, pady=16, anchor='w',
        ).pack(fill='x', pady=(0, 14))

        _primary(inner, '打开电脑微信', lambda: self._retry_wechat(note)).pack(fill='x')
        _ghost(inner, '再看一眼要发的文件', lambda: reveal_path(pack.folder)).pack(anchor='w', pady=(10, 0))
        note = tk.Label(inner, text='', bg=BG, fg=MUTED, font=('PingFang SC', 13))
        note.pack(anchor='w', pady=(8, 0))

        if urls:
            extra = tk.Frame(inner, bg=BG)

            def show_extra():
                extra.pack(fill='x', pady=(12, 0))

            _ghost(inner, '手机连着家里无线网？另一种办法', show_extra).pack(anchor='w', pady=(12, 0))
            tk.Label(
                extra,
                text='如果手机连的是家里的无线网，也可以在手机浏览器里打开：',
                bg=BG, fg=MUTED, font=('PingFang SC', 13), wraplength=520, justify='left',
            ).pack(anchor='w')
            tk.Label(
                extra, text='\n'.join(urls), bg=BG, fg=INK,
                font=('PingFang SC', 13), justify='left', wraplength=520,
            ).pack(anchor='w', pady=(6, 0))
            _ghost(extra, '抄下这个地址', lambda: self._copy(urls[0])).pack(anchor='w', pady=(8, 0))

    def _retry_wechat(self, label):
        if open_wechat():
            label.configure(text='微信已打开。去点「文件传输助手」，把文件拖进去。', fg=GREEN)
        else:
            label.configure(text='电脑上还没有微信。请先安装电脑版微信，登录后再点。', fg=RED)

    def _copy(self, text):
        if not text:
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update()
        self._set_status('已抄下来', GREEN)
        self.after(1600, lambda: self._set_status('找到了', GREEN) if not self._busy else None)

    def on_close(self):
        if self.proc and self.proc.poll() is None:
            if not messagebox.askyesno('要退出吗', '还在查找，现在退出就会停掉。确定退出？'):
                return
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.destroy()


def main():
    App().mainloop()


if __name__ == '__main__':
    main()
