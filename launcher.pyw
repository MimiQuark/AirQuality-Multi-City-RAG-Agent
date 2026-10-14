#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多城市空气质量监测 AI 系统 - GUI 启动器
Windows 双击运行（.pyw），无命令行黑框
功能：配置 API → 启动服务 → 自动打开浏览器
"""

import os
import sys
import json
import subprocess
import threading
import traceback
import webbrowser

# ───────────────────────────────────────────────
# 崩溃日志：pythonw 无控制台，任何异常都必须落盘，否则闪退无从排查
# ───────────────────────────────────────────────

ERROR_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "launcher_error.log")


def _write_crash_log(text):
    try:
        from datetime import datetime
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n===== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n{text}\n")
    except Exception:
        pass


def _excepthook(exc_type, exc, tb):
    _write_crash_log("".join(traceback.format_exception(exc_type, exc, tb)))


sys.excepthook = _excepthook

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, scrolledtext
except Exception:
    _write_crash_log("tkinter 导入失败：\n" + traceback.format_exc())
    raise

# ───────────────────────────────────────────────
# 配置管理
# ───────────────────────────────────────────────

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".launcher_config.json")

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_config(cfg):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

# ───────────────────────────────────────────────
# 主窗口
# ───────────────────────────────────────────────

class LauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("多城市空气质量监测 AI 系统 v2.1")
        self.root.geometry("560x640")
        self.root.resizable(False, False)
        self.root.configure(bg="#f5f7fa")

        # 样式
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f5f7fa")
        self.style.configure("TLabel", background="#f5f7fa", font=("Microsoft YaHei", 10))
        self.style.configure("TButton", font=("Microsoft YaHei", 10))
        self.style.configure("Header.TLabel", font=("Microsoft YaHei", 14, "bold"), foreground="#1976d2")
        self.style.configure("Sub.TLabel", font=("Microsoft YaHei", 9), foreground="#666")
        self.style.configure("Status.TLabel", font=("Microsoft YaHei", 10, "bold"))

        self.process = None
        self.config = load_config()

        self._build_ui()
        self._load_config_to_ui()

    def _build_ui(self):
        # 标题
        header = ttk.Label(self.root, text="🌐 多城市空气质量监测 AI 系统", style="Header.TLabel")
        header.pack(pady=(16, 4))
        sub = ttk.Label(self.root, text="LangGraph Agent + 时序预测 + RAG  |  零配置自动降级", style="Sub.TLabel")
        sub.pack(pady=(0, 12))

        # 主容器
        main = ttk.Frame(self.root, padding="20 10")
        main.pack(fill=tk.BOTH, expand=True)

        # ── LLM 配置 ──
        llm_frame = ttk.LabelFrame(main, text="🤖 大模型配置（可选）", padding="12 8")
        llm_frame.pack(fill=tk.X, pady=(0, 10))

        # 提供商选择
        ttk.Label(llm_frame, text="服务商:").grid(row=0, column=0, sticky=tk.W, pady=4)
        self.provider_var = tk.StringVar(value="deepseek")
        self.provider_combo = ttk.Combobox(llm_frame, textvariable=self.provider_var,
                                           values=["deepseek", "qwen", "openai", "custom"],
                                           state="readonly", width=14)
        self.provider_combo.grid(row=0, column=1, sticky=tk.W, padx=4, pady=4)
        self.provider_combo.bind("<<ComboboxSelected>>", self._on_provider_change)

        # API Key
        ttk.Label(llm_frame, text="API Key:").grid(row=1, column=0, sticky=tk.W, pady=4)
        self.api_key_var = tk.StringVar()
        self.api_key_entry = ttk.Entry(llm_frame, textvariable=self.api_key_var, width=38, show="•")
        self.api_key_entry.grid(row=1, column=1, sticky=tk.W, padx=4, pady=4)
        self.show_key_btn = ttk.Button(llm_frame, text="👁", width=3,
                                        command=self._toggle_key_visibility)
        self.show_key_btn.grid(row=1, column=2, padx=2)

        # 模型
        ttk.Label(llm_frame, text="模型:").grid(row=2, column=0, sticky=tk.W, pady=4)
        self.model_var = tk.StringVar(value="deepseek-chat")
        self.model_entry = ttk.Entry(llm_frame, textvariable=self.model_var, width=38)
        self.model_entry.grid(row=2, column=1, sticky=tk.W, padx=4, pady=4)

        # Base URL
        ttk.Label(llm_frame, text="Base URL:").grid(row=3, column=0, sticky=tk.W, pady=4)
        self.base_url_var = tk.StringVar(value="https://api.deepseek.com/v1")
        self.base_url_entry = ttk.Entry(llm_frame, textvariable=self.base_url_var, width=38)
        self.base_url_entry.grid(row=3, column=1, sticky=tk.W, padx=4, pady=4)

        # ── 高级配置 ──
        adv_frame = ttk.LabelFrame(main, text="⚙️ 高级配置", padding="12 8")
        adv_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(adv_frame, text="服务端口:").grid(row=0, column=0, sticky=tk.W, pady=4)
        self.port_var = tk.StringVar(value="8000")
        ttk.Entry(adv_frame, textvariable=self.port_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=4, pady=4)

        ttk.Label(adv_frame, text="WAQI Token (可选):").grid(row=1, column=0, sticky=tk.W, pady=4)
        self.waqi_var = tk.StringVar()
        ttk.Entry(adv_frame, textvariable=self.waqi_var, width=30).grid(row=1, column=1, sticky=tk.W, padx=4, pady=4)

        ttk.Label(adv_frame, text="和风天气 Key (可选):").grid(row=2, column=0, sticky=tk.W, pady=4)
        self.qweather_var = tk.StringVar()
        ttk.Entry(adv_frame, textvariable=self.qweather_var, width=30).grid(row=2, column=1, sticky=tk.W, padx=4, pady=4)

        self.auto_browser_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(adv_frame, text="启动后自动打开浏览器", variable=self.auto_browser_var).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=4)

        # ── 操作按钮 ──
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=(4, 8))

        self.start_btn = tk.Button(btn_frame, text="▶ 启动服务", bg="#4caf50", fg="white",
                                    font=("Microsoft YaHei", 11, "bold"),
                                    width=14, height=2, command=self._start_service,
                                    cursor="hand2", relief=tk.FLAT)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.stop_btn = tk.Button(btn_frame, text="⏹ 停止服务", bg="#f44336", fg="white",
                                   font=("Microsoft YaHei", 11, "bold"),
                                   width=14, height=2, command=self._stop_service,
                                   cursor="hand2", relief=tk.FLAT, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.test_btn = tk.Button(btn_frame, text="🧪 测试 Agent", bg="#2196f3", fg="white",
                                   font=("Microsoft YaHei", 11, "bold"),
                                   width=14, height=2, command=self._open_test,
                                   cursor="hand2", relief=tk.FLAT)
        self.test_btn.pack(side=tk.LEFT)

        # ── 状态栏 ──
        self.status_var = tk.StringVar(value="⏹ 服务未启动")
        status_bar = tk.Label(main, textvariable=self.status_var, bg="#e3f2fd",
                               fg="#1565c0", font=("Microsoft YaHei", 10, "bold"),
                               anchor=tk.W, padx=10, pady=6)
        status_bar.pack(fill=tk.X, pady=(4, 0))

        # ── 日志输出 ──
        log_frame = ttk.LabelFrame(main, text="📋 运行日志", padding="4 4")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10,
                                                    font=("Consolas", 9), state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # 底部说明
        footer = ttk.Label(self.root, text="未配置 API Key 时自动使用规则版 Agent（零依赖）", style="Sub.TLabel")
        footer.pack(pady=(4, 8))

    # ───────────────────────────────────────────────
    # 事件处理
    # ───────────────────────────────────────────────

    def _on_provider_change(self, event=None):
        provider = self.provider_var.get()
        presets = {
            "deepseek": ("https://api.deepseek.com/v1", "deepseek-chat"),
            "qwen":     ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-max"),
            "openai":   ("https://api.openai.com/v1", "gpt-4o"),
            "custom":   ("", ""),
        }
        url, model = presets.get(provider, ("", ""))
        self.base_url_var.set(url)
        self.model_var.set(model)

    def _toggle_key_visibility(self):
        if self.api_key_entry.cget("show") == "•":
            self.api_key_entry.configure(show="")
            self.show_key_btn.configure(text="🙈")
        else:
            self.api_key_entry.configure(show="•")
            self.show_key_btn.configure(text="👁")

    def _load_config_to_ui(self):
        self.provider_var.set(self.config.get("provider", "deepseek"))
        self.api_key_var.set(self.config.get("api_key", ""))
        self.model_var.set(self.config.get("model", "deepseek-chat"))
        self.base_url_var.set(self.config.get("base_url", "https://api.deepseek.com/v1"))
        self.port_var.set(str(self.config.get("port", 8000)))
        self.waqi_var.set(self.config.get("waqi_token", ""))
        self.qweather_var.set(self.config.get("qweather_key", ""))
        self.auto_browser_var.set(self.config.get("auto_browser", True))

    def _save_ui_to_config(self):
        self.config = {
            "provider": self.provider_var.get(),
            "api_key": self.api_key_var.get().strip(),
            "model": self.model_var.get().strip(),
            "base_url": self.base_url_var.get().strip(),
            "port": int(self.port_var.get() or 8000),
            "waqi_token": self.waqi_var.get().strip(),
            "qweather_key": self.qweather_var.get().strip(),
            "auto_browser": self.auto_browser_var.get(),
        }
        save_config(self.config)

    def _log(self, msg):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    # ───────────────────────────────────────────────
    # 服务控制
    # ───────────────────────────────────────────────

    def _probe_port(self, port):
        """探测端口状态: 'ours'=本系统已在运行 / 'occupied'=被其他程序占用 / 'free'=空闲"""
        import urllib.request
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2) as r:
                if r.status == 200:
                    return "ours"
        except urllib.error.HTTPError:
            return "occupied"   # 有 HTTP 服务但不是本系统
        except Exception:
            pass
        # 无 HTTP 响应，进一步确认端口是否被非 HTTP 程序占用
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        try:
            s.connect(("127.0.0.1", int(port)))
            return "occupied"
        except Exception:
            return "free"
        finally:
            s.close()

    def _find_port_pid(self, port):
        """Windows 下通过 netstat 查找占用指定端口的 PID"""
        try:
            out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True,
                                 encoding="utf-8", errors="replace").stdout
            for line in out.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    addr = parts[1]
                    if addr.endswith(f":{port}"):
                        return parts[-1]
        except Exception:
            pass
        return None

    def _start_service(self):
        if self.process is not None:
            messagebox.showwarning("提示", "服务已在运行中")
            return

        self._save_ui_to_config()
        port = self.config["port"]

        # 端口预检：服务已在运行则直接接管，被其他程序占用则明确报错
        state = self._probe_port(port)
        if state == "ours":
            self._log(f"[{self._now()}] ✅ 检测到服务已在运行（端口 {port}），直接打开浏览器")
            self._update_ui_running(True)
            if self.auto_browser_var.get():
                webbrowser.open(f"http://127.0.0.1:{port}")
            return
        if state == "occupied":
            pid = self._find_port_pid(port)
            tip = f"端口 {port} 已被其他程序占用" + (f"（PID {pid}）" if pid else "")
            self._log(f"[{self._now()}] ❌ {tip}")
            messagebox.showerror("端口被占用", f"{tip}\n\n请换一个端口，或在命令行执行：\ntaskkill /F /PID {pid or '<PID>'}")
            return

        # 环境变量
        env = os.environ.copy()
        if self.config["api_key"]:
            env["LLM_API_KEY"] = self.config["api_key"]
            env["LLM_BASE_URL"] = self.config["base_url"]
            env["LLM_MODEL"] = self.config["model"]
        if self.config.get("waqi_token"):
            env["WAQI_TOKEN"] = self.config["waqi_token"]
        if self.config.get("qweather_key"):
            env["QWEATHER_KEY"] = self.config["qweather_key"]

        # 项目目录
        project_dir = os.path.dirname(os.path.abspath(__file__))

        self._log(f"[{self._now()}] 正在启动服务...")
        self._log(f"[{self._now()}] 端口: {port}")
        self._log(f"[{self._now()}] 目录: {project_dir}")

        try:
            self.process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "backend.main:app",
                 "--host", "127.0.0.1", "--port", str(port)],
                cwd=project_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        except Exception as e:
            messagebox.showerror("启动失败", str(e))
            self._log(f"[{self._now()}] ❌ 启动失败: {e}")
            return

        # 早期失败检测：进程若在 3 秒内退出，说明启动报错（依赖缺失/端口占用等）
        import time as _time
        _time.sleep(3)
        if self.process.poll() is not None:
            # 进程已退出：把残留的错误输出全部读出来显示，方便定位问题
            try:
                leftover = self.process.stdout.read() if self.process.stdout else ""
                for line in leftover.splitlines():
                    if line.strip():
                        self._log(line.strip())
            except Exception:
                pass
            self._log(f"[{self._now()}] ❌ 服务进程已退出（代码 {self.process.returncode}），请查看上方日志")
            messagebox.showerror("启动失败", "服务进程启动后立即退出，请查看日志输出。\n常见原因：缺少依赖（请先 pip install -r requirements.txt）或端口被占用。")
            self.process = None
            return

        self._update_ui_running(True)

        # 日志读取线程
        threading.Thread(target=self._read_output, daemon=True).start()

        # 等待服务就绪 + 自动打开浏览器
        if self.auto_browser_var.get():
            threading.Thread(target=self._wait_and_open, args=(port,), daemon=True).start()

    def _read_output(self):
        if self.process and self.process.stdout:
            for line in iter(self.process.stdout.readline, ''):
                if not line:
                    break
                self.root.after(0, lambda l=line.strip(): self._log(l))

    def _wait_and_open(self, port):
        import time
        import urllib.request
        url = f"http://127.0.0.1:{port}"
        self._log(f"[{self._now()}] 等待服务就绪...")
        for _ in range(90):  # 最多等 90 秒（LLM 模式首次加载 langchain 较慢）
            time.sleep(1)
            try:
                urllib.request.urlopen(url, timeout=2)
                self.root.after(0, lambda: self._log(f"[{self._now()}] ✅ 服务已就绪"))
                self.root.after(0, lambda: self._log(f"[{self._now()}] 🚀 正在打开浏览器..."))
                webbrowser.open(url)
                return
            except Exception:
                pass
        self.root.after(0, lambda: self._log(f"[{self._now()}] ⚠️ 服务启动超时，请手动访问 {url}"))

    def _stop_service(self):
        if self.process:
            self._log(f"[{self._now()}] 正在停止服务...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self._log(f"[{self._now()}] 强制终止服务")
            self.process = None
            self._update_ui_running(False)
            self._log(f"[{self._now()}] ⏹ 服务已停止")
            return

        # 接管模式：进程不是本 GUI 启动的（如之前遗留的服务），按端口找 PID 后结束
        port = self.config.get("port", 8000)
        if self._probe_port(port) == "ours":
            pid = self._find_port_pid(port)
            if pid:
                self._log(f"[{self._now()}] 正在停止端口 {port} 上的服务（PID {pid}）...")
                subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
                self._update_ui_running(False)
                self._log(f"[{self._now()}] ⏹ 服务已停止")
                return
        self._log(f"[{self._now()}] 没有正在运行的服务")

    def _open_test(self):
        # 直达前端 AI 问答页签实测 Agent（/api/agent/tools 为原始 JSON，仅供开发查看）
        port = self.config.get("port", 8000)
        webbrowser.open(f"http://127.0.0.1:{port}/#chat")

    def _update_ui_running(self, running):
        if running:
            self.start_btn.configure(state=tk.DISABLED, bg="#81c784")
            self.stop_btn.configure(state=tk.NORMAL)
            self.status_var.set(f"🟢 服务运行中 | http://127.0.0.1:{self.config['port']}")
        else:
            self.start_btn.configure(state=tk.NORMAL, bg="#4caf50")
            self.stop_btn.configure(state=tk.DISABLED)
            self.status_var.set("⏹ 服务未启动")

    def _now(self):
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")

    def on_close(self):
        self._stop_service()
        self.root.destroy()


# ───────────────────────────────────────────────
# 快速模式（命令行，无 GUI）
# ───────────────────────────────────────────────

def quick_start():
    """命令行快速启动：读取上次保存的配置直接启动服务"""
    cfg = load_config()
    port = cfg.get("port", 8000)

    env = os.environ.copy()
    if cfg.get("api_key"):
        env["LLM_API_KEY"] = cfg["api_key"]
        env["LLM_BASE_URL"] = cfg.get("base_url", "https://api.deepseek.com/v1")
        env["LLM_MODEL"] = cfg.get("model", "deepseek-chat")
    if cfg.get("waqi_token"):
        env["WAQI_TOKEN"] = cfg["waqi_token"]
    if cfg.get("qweather_key"):
        env["QWEATHER_KEY"] = cfg["qweather_key"]

    project_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"🚀 快速启动模式 | 端口: {port}")
    print(f"   配置来源: {CONFIG_FILE}")
    print(f"   按 Ctrl+C 停止\n")

    subprocess.run(
        [sys.executable, "-m", "uvicorn", "backend.main:app",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=project_dir,
        env=env,
    )


# ───────────────────────────────────────────────
# 入口
# ───────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--quick", "-q"):
        quick_start()
    else:
        try:
            root = tk.Tk()
            # tkinter 回调里的异常默认只打到 stderr（pythonw 下不可见），统一落盘
            root.report_callback_exception = lambda t, e, tb: _write_crash_log(
                "".join(traceback.format_exception(t, e, tb)))
            app = LauncherApp(root)
            root.protocol("WM_DELETE_WINDOW", app.on_close)
            root.mainloop()
        except Exception:
            err = traceback.format_exc()
            _write_crash_log(err)
            try:
                _r = tk.Tk()
                _r.withdraw()
                messagebox.showerror("启动器错误", err[-600:])
            except Exception:
                pass
