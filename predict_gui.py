# 导入必要的模块
import os
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk
import numpy as np
import torch
import torch.nn.functional as F
from train_mlp import MLP, HIDDEN_SIZE, DEVICE
from scipy.ndimage import zoom
import argparse

# 可选导入 PIL 库，用于图像处理
try:
    from PIL import Image

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# --------------- 配置常量 ----------------
GRID_SIZE = 28  # 28x28 网格
CELL_SIZE = 20  # 画布中每个单元格的像素大小
CANVAS_SIZE = GRID_SIZE * CELL_SIZE
MODEL_PATH = 'best_mlp_mnist.pth'
BRUSH_RADIUS = 24  # 画笔半径，用于平滑绘制（粗三倍）


class PredictApp:
    """一个 Tkinter 应用程序，用于手写数字绘制、模拟 MNIST 预处理、
    预测和可视化。背景黑色，笔白色，平滑绘制。
    """

    def __init__(self, master):
        self.master = master
        master.title('MNIST 手写数字预测器')

        # 逻辑网格存储 0.0（黑色背景）或 1.0（白色墨迹）
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # 布局：画布在上，按钮在画布下方，右侧控件
        self.canvas_frame = tk.Frame(master)
        self.canvas_frame.grid(row=0, column=0, sticky='nsew')
        self.button_frame = tk.Frame(master)
        self.button_frame.grid(row=1, column=0, sticky='ew')
        self.right_frame = tk.Frame(master)
        self.right_frame.grid(row=0, column=1, sticky='nsew')
        master.grid_columnconfigure(0, weight=1)
        master.grid_columnconfigure(1, weight=0)
        master.grid_rowconfigure(0, weight=1)

        # 绘制画布（黑色背景）
        self.canvas = tk.Canvas(self.canvas_frame, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='black')
        self.canvas.pack(padx=10, pady=10)

        # 绑定鼠标事件用于绘制
        self.canvas.bind('<Button-1>', self._on_paint_start)
        self.canvas.bind('<B1-Motion>', self._on_paint_motion)
        self.canvas.bind('<ButtonRelease-1>', self._on_paint_release)

        # 按钮放在画布下方，分两行排列
        for col in range(2):
            self.button_frame.grid_columnconfigure(col, weight=1)

        self.predict_btn = tk.Button(self.button_frame, text='预测', command=self.predict)
        self.predict_btn.grid(row=0, column=0, sticky='ew', padx=6, pady=4)

        self.clear_btn = tk.Button(self.button_frame, text='清空', command=self.clear_grid)
        self.clear_btn.grid(row=0, column=1, sticky='ew', padx=6, pady=4)

        self.load_btn = tk.Button(self.button_frame, text='加载模型', command=self.load_model_dialog)
        self.load_btn.grid(row=1, column=0, sticky='ew', padx=6, pady=4)

        self.save_btn = tk.Button(self.button_frame, text='保存图像', command=self.save_image)
        self.save_btn.grid(row=1, column=1, sticky='ew', padx=6, pady=4)

        # 右侧框架的控件
        # 预测显示
        self.result_label = tk.Label(self.right_frame, text='预测: -', font=('Helvetica', 14))
        self.result_label.pack(anchor='w', padx=6, pady=(6, 0))

        # 所有类别的概率条
        self.prob_frame = ttk.LabelFrame(self.right_frame, text="各类别概率", padding=10)
        self.prob_frame.pack(padx=6, pady=6, fill='x')

        self.prob_bars = []
        self.prob_labels = []
        for i in range(10):
            frame = ttk.Frame(self.prob_frame)
            frame.pack(fill='x', pady=2)

            label = tk.Label(frame, text=str(i), width=2, anchor='w')
            label.pack(side='left')

            bar = ttk.Progressbar(frame, length=150, mode='determinate')
            bar.pack(side='left', padx=5)

            value_label = tk.Label(frame, text="0.0%", width=6, anchor='w')
            value_label.pack(side='left')

            self.prob_bars.append(bar)
            self.prob_labels.append(value_label)

        # 实际 28x28 模型输入的预览（放大）
        PREVIEW_SCALE = 8
        self.preview_scale = PREVIEW_SCALE
        self.preview_canvas = tk.Canvas(self.right_frame, width=GRID_SIZE * PREVIEW_SCALE,
                                        height=GRID_SIZE * PREVIEW_SCALE, bg='black', bd=1, relief='sunken')
        self.preview_canvas.pack(padx=6, pady=6)
        self.preview_photo = None

        # 模型占位符
        self.model = None
        self._try_load_default_model()

        # 绘制状态
        self.drawing = False
        self.last_x = None
        self.last_y = None

        # 初始预览
        self._update_preview()

    def _on_paint_start(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
        self._draw_brush(event.x, event.y)

    def _on_paint_motion(self, event):
        if self.drawing:
            self._draw_line(self.last_x, self.last_y, event.x, event.y)
            self.last_x = event.x
            self.last_y = event.y

    def _on_paint_release(self, event):
        self.drawing = False
        self.last_x = None
        self.last_y = None
        # 完成绘制后更新预览
        self._update_preview()

    def _draw_brush(self, x, y):
        # 绘制圆形笔刷，白色
        self.canvas.create_oval(x - BRUSH_RADIUS, y - BRUSH_RADIUS, x + BRUSH_RADIUS, y + BRUSH_RADIUS, fill='white',
                                outline='')

        # 更新网格：设置覆盖的单元为 1.0
        left = max(0, int((x - BRUSH_RADIUS) // CELL_SIZE))
        right = min(GRID_SIZE - 1, int((x + BRUSH_RADIUS) // CELL_SIZE))
        top = max(0, int((y - BRUSH_RADIUS) // CELL_SIZE))
        bottom = min(GRID_SIZE - 1, int((y + BRUSH_RADIUS) // CELL_SIZE))
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                self.grid[r, c] = 1.0

    def _draw_line(self, x1, y1, x2, y2):
        # 绘制平滑线条，通过插值点
        dist = max(abs(x2 - x1), abs(y2 - y1))
        steps = int(dist / 2) + 1
        for i in range(steps + 1):
            t = i / steps
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            self._draw_brush(x, y)

    def clear_grid(self):
        self.canvas.delete('all')
        self.grid.fill(0.0)
        self.result_label.config(text='预测: -')
        for bar, label in zip(self.prob_bars, self.prob_labels):
            bar['value'] = 0
            label.config(text="0.0%")
        # 清空后更新预览
        self._update_preview()

    def _set_probs(self, probs):
        for i, (bar, label) in enumerate(zip(self.prob_bars, self.prob_labels)):
            p = probs[i] * 100
            bar['value'] = p
            label.config(text=f"{p:.1f}%")

    def _try_load_default_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.model = self._load_model(MODEL_PATH)
                print(f'从 {MODEL_PATH} 加载模型')
            except Exception as e:
                messagebox.showwarning('模型加载警告', f'加载默认模型失败: {e}')
        else:
            print('未找到默认模型；使用 "加载模型..." 选择 .pth 文件')

    def load_model_dialog(self):
        p = filedialog.askopenfilename(title='选择模型文件', filetypes=[('PyTorch', '*.pth'), ('所有文件', '*.*')])
        if not p:
            return
        try:
            self.model = self._load_model(p)
            messagebox.showinfo('模型已加载', f'从 {p} 加载模型')
        except Exception as e:
            messagebox.showerror('加载失败', f'加载模型失败:\n{e}')

    def _load_model(self, path):
        # 实例化与训练相同的 MLP 架构
        model = MLP(GRID_SIZE * GRID_SIZE, HIDDEN_SIZE, 10)
        # 加载状态字典；映射到 CPU 或 DEVICE（如果可用）
        map_location = torch.device('cpu') if not torch.cuda.is_available() else DEVICE
        state = torch.load(path, map_location=map_location)
        # 支持 state_dict 和整个模型保存
        if isinstance(state, dict) and 'state_dict' in state and not any(k.startswith('fc') for k in state.keys()):
            model.load_state_dict(state['state_dict'])
        else:
            try:
                model.load_state_dict(state)
            except Exception:
                model = state
        model.to(map_location)
        model.eval()
        return model

    def _preprocess_for_mnist(self, img):
        """模拟 MNIST 预处理：裁剪最小包围框，缩放到 ~20x20，保持比例，居中填充到 28x28，保留灰度。"""
        # 找到非零像素的包围框
        rows = np.any(img > 0, axis=1)
        cols = np.any(img > 0, axis=0)
        if not np.any(rows) or not np.any(cols):
            return np.zeros((28, 28), dtype=np.float32)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        cropped = img[rmin:rmax + 1, cmin:cmax + 1]

        # 缩放到 ~20x20，保持比例
        h, w = cropped.shape
        scale = min(20 / h, 20 / w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        resized = zoom(cropped, (new_h / h, new_w / w), order=1)  # 线性插值，抗锯齿

        # 居中放置在 28x28 中，加 4 像素边距
        final = np.zeros((28, 28), dtype=np.float32)
        offset_h = (28 - new_h) // 2
        offset_w = (28 - new_w) // 2
        final[offset_h:offset_h + new_h, offset_w:offset_w + new_w] = resized

        return final

    def _preprocess_grid(self):
        """将逻辑网格转换为模型输入，应用 MNIST 预处理。"""
        # 从画布获取图像（简化：假设网格已更新，但实际需要从画布读取）
        # 这里简化，使用 self.grid，但实际应从画布像素读取
        img = self.grid.copy()
        processed = self._preprocess_for_mnist(img)
        tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0).float()
        return tensor

    def predict(self):
        if self.model is None:
            messagebox.showerror('无模型', '未加载模型。请先加载模型。')
            return
        try:
            x = self._preprocess_grid()
            device = next(self.model.parameters()).device
            x = x.to(device)
            with torch.no_grad():
                logits = self.model(x)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                pred = int(np.argmax(probs))

            self.result_label.config(text=f'预测: {pred} (置信度 {probs[pred]:.2f})')
            self._set_probs(probs)

        except Exception as e:
            messagebox.showerror('预测错误', f'预测期间错误:\n{e}')

    def save_image(self):
        # 保存预处理后的 28x28 图像
        img = self._get_model_input_np() * 255
        img = img.astype(np.uint8)
        if PIL_AVAILABLE:
            im = Image.fromarray(img, mode='L')
            p = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG', '*.png')])
            if not p:
                return
            im.save(p)
            messagebox.showinfo('已保存', f'图像保存到 {p}')
        else:
            p = filedialog.asksaveasfilename(defaultextension='.pgm', filetypes=[('PGM', '*.pgm'), ('所有文件', '*.*')])
            if not p:
                return
            with open(p, 'wb') as f:
                header = f'P5\n{GRID_SIZE} {GRID_SIZE}\n255\n'
                f.write(header.encode('ascii'))
                f.write(img.tobytes())
            messagebox.showinfo('已保存', f'图像保存到 {p} (PGM 格式)')

    def _get_model_input_np(self):
        """返回预处理后的 28x28 numpy 数组。"""
        t = self._preprocess_grid()
        return t.squeeze().cpu().numpy()

    def _update_preview(self):
        """更新预览画布显示预处理后的图像。"""
        arr = self._get_model_input_np()
        arr = np.clip(arr, 0.0, 1.0)
        s = self.preview_scale
        self.preview_canvas.delete('all')
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                v = int(arr[r, c] * 255)
                hexc = f'#{v:02x}{v:02x}{v:02x}'
                x1 = c * s
                y1 = r * s
                x2 = x1 + s
                y2 = y1 + s
                self.preview_canvas.create_rectangle(x1, y1, x2, y2, fill=hexc, outline=hexc)

        # 绘制网格
        for i in range(0, GRID_SIZE * s, s):
            self.preview_canvas.create_line(i, 0, i, GRID_SIZE * s, fill='#111111')
            self.preview_canvas.create_line(0, i, GRID_SIZE * s, i, fill='#111111')


def load_image_as_array(path):
    """从文件加载图像并转换为 28x28 灰度 numpy 数组。"""
    if not PIL_AVAILABLE:
        raise ValueError("PIL not available, cannot load image")
    img = Image.open(path).convert('L')  # 转换为灰度
    img = img.resize((28, 28), Image.Resampling.LANCZOS)  # 缩放到 28x28
    arr = np.array(img, dtype=np.float32) / 255.0  # 归一化到 [0,1]
    return arr


def predict_from_image(image_path, model_path=MODEL_PATH):
    """从图像文件预测数字。"""
    # 加载模型
    model = MLP(GRID_SIZE * GRID_SIZE, HIDDEN_SIZE, 10)
    map_location = torch.device('cpu') if not torch.cuda.is_available() else DEVICE
    state = torch.load(model_path, map_location=map_location)
    if isinstance(state, dict) and 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    else:
        model.load_state_dict(state)
    model.to(map_location)
    model.eval()

    # 加载图像
    img = load_image_as_array(image_path)
    # 预处理
    processed = _preprocess_for_mnist_static(img)
    tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0).float()
    device = next(model.parameters()).device
    tensor = tensor.to(device)

    # 预测
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))

    print(f'预测结果: {pred}')
    print('概率:')
    for i, p in enumerate(probs):
        print(f'{i}: {p:.4f}')


def _preprocess_for_mnist_static(img):
    """静态版本的 MNIST 预处理函数。"""
    # 找到非零像素的包围框
    rows = np.any(img > 0, axis=1)
    cols = np.any(img > 0, axis=0)
    if not np.any(rows) or not np.any(cols):
        return np.zeros((28, 28), dtype=np.float32)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = img[rmin:rmax + 1, cmin:cmax + 1]

    # 缩放到 ~20x20，保持比例
    h, w = cropped.shape
    scale = min(20 / h, 20 / w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    resized = zoom(cropped, (new_h / h, new_w / w), order=1)  # 线性插值，抗锯齿

    # 居中放置在 28x28 中，加 4 像素边距
    final = np.zeros((28, 28), dtype=np.float32)
    offset_h = (28 - new_h) // 2
    offset_w = (28 - new_w) // 2
    final[offset_h:offset_h + new_h, offset_w:offset_w + new_w] = resized

    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST 手写数字预测器')
    parser.add_argument('--image', type=str, help='图像文件路径，用于命令行预测')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='模型文件路径')
    args = parser.parse_args()

    if args.image:
        # 命令行预测
        predict_from_image(args.image, args.model)
    else:
        # 启动 GUI
        root = tk.Tk()
        app = PredictApp(root)
        root.mainloop()
