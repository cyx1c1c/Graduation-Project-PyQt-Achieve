from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QTabWidget, \
    QHeaderView, QTextEdit
from PyQt5.QtGui import QFont, QColor, QBrush, QPixmap
from PyQt5.QtWidgets import QScrollArea
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import sys
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
import os
import shutil
from PyQt5.QtWidgets import (QPushButton, QFileDialog, QHBoxLayout, QFrame, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import lpips
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as T

from model import DecomNet, RelightNet

# 权重配置
WEIGHTS = {
    "LPIPS": 0.317,
    "SSIM": 0.264,
    "GCF": 0.192,
    "PSNR": 0.146,
    "MAE": 0.049,
    "STD": 0.032
}

# 解决 Matplotlib 中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 文件路径
pyqt_initial_files = "results/results_014.json"
pyqt_files = "results/results_017.json"


def compute_metrics(original, enhanced):
    lpips_model = lpips.LPIPS(net='alex').to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    """计算 PSNR、SSIM、LPIPS、MAE、STD、GCF 评价指标"""
    original_np = np.array(original) / 255.0
    enhanced_np = np.array(enhanced) / 255.0

    # 计算各项指标
    psnr_value = psnr(original_np, enhanced_np, data_range=1.0)
    # 确保 win_size 不超过图像的最小边长
    min_dim = min(original_np.shape[0], original_np.shape[1])
    win_size = min(7, min_dim)
    ssim_value = ssim(original_np, enhanced_np, data_range=1.0, channel_axis=-1, win_size=win_size)
    lpips_value = lpips_model(
        torch.tensor(original_np).permute(2, 0, 1).unsqueeze(0).float(),
        torch.tensor(enhanced_np).permute(2, 0, 1).unsqueeze(0).float()
    ).item()
    mae_value = np.mean(np.abs(original_np - enhanced_np))
    std_value = np.std(enhanced_np)
    gcf_value = np.mean(np.abs(enhanced_np - enhanced_np.mean()))

    return {"PSNR": psnr_value, "SSIM": ssim_value, "LPIPS": lpips_value, "MAE": mae_value, "STD": std_value,
            "GCF": gcf_value}


class ProcessingThread(QThread):
    progress_updated = pyqtSignal(int)
    processing_finished = pyqtSignal(str, str, dict)

    def __init__(self, input_path, ckpt_dir):
        super().__init__()
        self.input_path = input_path
        self.ckpt_dir = ckpt_dir

    def run(self):
        self.progress_updated.emit(10)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型
        decom_net = DecomNet().to(device)
        relight_net = RelightNet().to(device)

        # 加载 Decom 权重
        decom_ckpt_dir = "ckpts/Decom"
        decom_ckpt_list = sorted(os.listdir(decom_ckpt_dir), key=lambda x: int(x.replace(".tar", "").strip()))
        if not decom_ckpt_list:
            print("未找到 Decom 模型，请确认 ckpts/Decom 中有 .tar 文件")
            return
        decom_ckpt_path = os.path.join(decom_ckpt_dir, decom_ckpt_list[-1])
        decom_state_dict = torch.load(decom_ckpt_path, map_location=device)
        decom_net.load_state_dict(decom_state_dict)

        # 加载 Relight 权重
        relight_ckpt_list = sorted(os.listdir(self.ckpt_dir), key=lambda x: int(x.replace(".tar", "").strip()))
        if not relight_ckpt_list:
            print("未找到 Relight 模型，请确认 ckpts/Relight 中有 .tar 文件")
            return
        relight_ckpt_path = os.path.join(self.ckpt_dir, relight_ckpt_list[-1])
        relight_state_dict = torch.load(relight_ckpt_path, map_location=device)
        relight_net.load_state_dict(relight_state_dict)

        decom_net.eval()
        relight_net.eval()
        self.progress_updated.emit(30)

        # 加载图像并预处理
        img = Image.open(self.input_path).convert("RGB")
        transform = T.Compose([
            T.ToTensor(),  # [0,1]
        ])
        input_tensor = transform(img).unsqueeze(0).to(device)

        # 分解
        with torch.no_grad():
            R, L = decom_net(input_tensor)
            I_delta = relight_net(L, R)
            I_delta_3 = torch.cat([I_delta] * 3, dim=1)
            enhanced = R * I_delta_3
            enhanced_img = enhanced.squeeze().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255
            enhanced_img = Image.fromarray(enhanced_img.astype(np.uint8))

        # 保存输出图像
        output_dir = "image_showing/output"
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(self.input_path)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_Enhanced.jpg")
        enhanced_img.save(output_path)

        self.progress_updated.emit(70)

        # 评价指标
        original_np = np.array(img).astype(np.float32) / 255.0
        enhanced_np = np.array(enhanced_img).astype(np.float32) / 255.0
        min_dim = min(original_np.shape[0], original_np.shape[1])
        win_size = min(7, min_dim)

        lpips_model = lpips.LPIPS(net='alex').to(device)

        metrics = {
            "PSNR": psnr(original_np, enhanced_np, data_range=1.0),
            "SSIM": ssim(original_np, enhanced_np, data_range=1.0, channel_axis=-1, win_size=win_size),
            "LPIPS": lpips_model(
                torch.tensor(original_np).permute(2, 0, 1).unsqueeze(0).float().to(device),
                torch.tensor(enhanced_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
            ).item(),
            "MAE": np.mean(np.abs(original_np - enhanced_np)),
            "STD": np.std(enhanced_np),
            "GCF": np.mean(np.abs(enhanced_np - enhanced_np.mean()))
        }

        self.progress_updated.emit(100)
        self.processing_finished.emit(self.input_path, output_path, metrics)


class ImageProcessingWidget(QWidget):
    theme_changed = pyqtSignal(bool)  # True=dark, False=light

    def __init__(self):
        super().__init__()
        self.current_file = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # 上传按钮
        self.upload_btn = QPushButton("上传图片")
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background: #3498DB;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 4px;
                font-family: Microsoft YaHei;
            }
            QPushButton:hover { background: #2980B9; }
        """)
        self.upload_btn.clicked.connect(self.upload_image)
        layout.addWidget(self.upload_btn, 0, Qt.AlignCenter)

        export_layout = QHBoxLayout()
        export_layout.addStretch()  # 靠右对齐

        # 导出评分按钮
        self.export_btn = QPushButton("导出评分")
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ECC71;
                color: white;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #27AE60;
            }
        """)
        self.export_btn.clicked.connect(self.export_metrics)
        self.export_btn.hide()
        export_layout.addWidget(self.export_btn)

        # 导出图片按钮
        self.export_img_btn = QPushButton("导出图片")
        self.export_img_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ECC71;
                color: white;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #27AE60;
            }
        """)
        self.export_img_btn.clicked.connect(self.export_image)
        self.export_img_btn.hide()
        export_layout.addWidget(self.export_img_btn)

        layout.addLayout(export_layout)

        # 进度条
        self.progress = QProgressBar()
        self.progress.setStyleSheet("""
            QProgressBar {
                height: 20px;
                text-align: center;
                border: 1px solid #BDC3C7;
                border-radius: 8px;
            }
            QProgressBar::chunk {
                background: #3498DB;
                border-radius: 8px;
            }
        """)
        self.progress.hide()
        layout.addWidget(self.progress)

        # 图片 + 表格容器
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QHBoxLayout(scroll_widget)  # 水平排列：左表格，右图片
        scroll_layout.setContentsMargins(10, 10, 10, 10)
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

        # 评分表格
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setRowCount(7)
        self.metrics_table.setHorizontalHeaderLabels(["指标", "值"])
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metrics_table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 1px solid #BDC3C7;
                border-radius: 6px;
                font-family: Microsoft YaHei;
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #3498DB;
                color: white;
            }
        """)
        self.metrics_table.hide()

        # 图片显示
        self.enhanced_label = QLabel()
        self.enhanced_label.setAlignment(Qt.AlignCenter)
        self.enhanced_label.setScaledContents(True)
        self.enhanced_label.setMinimumSize(1, 1)
        self.enhanced_label.setMaximumSize(16777215, 800)

        # 添加两个组件
        scroll_layout.addWidget(self.metrics_table, 1)
        scroll_layout.addWidget(self.enhanced_label, 4)

        self.setLayout(layout)

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png)"
        )
        if file_path:
            input_dir = "image_showing/input"
            os.makedirs(input_dir, exist_ok=True)
            filename = os.path.basename(file_path)
            save_path = os.path.join(input_dir, filename)
            shutil.copy(file_path, save_path)
            self.start_processing(save_path)

    def export_metrics(self):
        if not hasattr(self, "latest_metrics_text"):
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "保存评分", "", "Text Files (*.txt)")
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(self.latest_metrics_text)

    def export_image(self):
        if hasattr(self, 'enhanced_image_path') and os.path.exists(self.enhanced_image_path):
            save_path, _ = QFileDialog.getSaveFileName(self, "保存增强图像", "",
                                                       "JPEG Image (*.jpg *.jpeg);;PNG Image (*.png)")
            if save_path:
                shutil.copy(self.enhanced_image_path, save_path)

    def start_processing(self, input_path):
        self.current_file = input_path
        self.progress.show()
        self.progress.setValue(0)

        self.thread = ProcessingThread(input_path, "ckpts/Relight")
        self.thread.progress_updated.connect(self.update_progress)
        self.thread.processing_finished.connect(self.show_results)
        self.thread.start()

    def update_progress(self, value):
        self.progress.setValue(value)

    def show_results(self, input_path, output_path, metrics):
        score_new = sum(metrics[k] * WEIGHTS[k] for k in WEIGHTS)

        # 填入表格
        rows = [
            ("综合评分", f"{score_new:.4f}"),
            ("PSNR", f"{metrics['PSNR']:.2f}"),
            ("SSIM", f"{metrics['SSIM']:.4f}"),
            ("LPIPS", f"{metrics['LPIPS']:.4f}"),
            ("MAE", f"{metrics['MAE']:.4f}"),
            ("STD", f"{metrics['STD']:.4f}"),
            ("GCF", f"{metrics['GCF']:.4f}")
        ]
        self.latest_metrics_text = "\n".join(f"{k}: {v}" for k, v in rows)

        for i, (label, value) in enumerate(rows):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(label))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(value))

        self.metrics_table.show()
        self.export_btn.show()
        self.export_img_btn.show()
        self.enhanced_image_path = output_path
        self.show_image(output_path)

    def show_image(self, enhanced_path):
        original = Image.open(self.current_file)
        enhanced = Image.open(enhanced_path)

        # 拼接图像：左原图，右增强图
        w, h = original.size
        result_img = Image.new("RGB", (w * 2, h))
        result_img.paste(original, (0, 0))
        result_img.paste(enhanced, (w, 0))

        result_path = "image_showing/temp_combined.jpg"
        result_img.save(result_path)

        pixmap = QPixmap(result_path)
        self.enhanced_label.setPixmap(pixmap)


class MetricsWindow(QWidget):
    def __init__(self, initial_results, new_results):
        super().__init__()
        self.setWindowTitle("图像增强综合评价系统")
        self.setGeometry(100, 100, 1000, 800)
        self.setStyleSheet("background-color: #F5F5F5;")

        # 主布局
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # 标题
        title = QLabel("✨ 图像增强综合评价平台 ✨")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                color: #2C3E50;
                margin-bottom: 20px;
                border-bottom: 2px solid #3498DB;
                padding-bottom: 10px;
            }
        """)
        layout.addWidget(title)

        # 添加切换深色模式按钮
        self.dark_mode = False  # 默认浅色
        self.theme_btn = QPushButton("切换深色/浅色模式")
        self.theme_btn.setStyleSheet("""
            QPushButton {
                background-color: #34495E;
                color: white;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #2C3E50;
            }
        """)
        self.theme_btn.clicked.connect(self.toggle_theme)
        layout.addWidget(self.theme_btn, 0, Qt.AlignRight)

        # Tab 界面
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabBar::tab {
                min-width: 120px;
                height: 32px;
                padding: 10px 24px;
                background: #ECF0F1;
                border: 1px solid #BDC3C7;
                border-radius: 6px;
                font-family: Microsoft YaHei;
                font-size: 13px;
            }
            QTabBar::tab:selected {
                background: #3498DB;
                color: white;
                font-weight: bold;
            }
            QTabWidget::pane {
                border-top: 2px solid #3498DB;
                background: #FFFFFF;
            }
        """)

        # 创建界面组件
        self.table_widget = self.create_table_widget(initial_results, new_results)
        self.plot_widget = self.create_plot_widget(initial_results, new_results)
        self.imageProcessing_widget = self.create_imageProcessing_widget()

        self.tabs.addTab(self.table_widget, "综合评分对比")
        self.tabs.addTab(self.plot_widget, "评分趋势分析")
        self.tabs.addTab(self.imageProcessing_widget, "在线增强处理")

        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        if self.dark_mode:
            self.apply_dark_theme()
        else:
            self.apply_light_theme()

    def apply_dark_theme(self):
        self.setStyleSheet("background-color: #2C3E50; color: white;")
        self.findChild(QLabel).setStyleSheet("""
            QLabel {
                color: white;
                margin-bottom: 20px;
                border-bottom: 2px solid #3498DB;
                padding-bottom: 10px;
            }
        """)
        self.processing_widget.metrics_table.setStyleSheet("""
            QTableWidget {
                background-color: #34495E;
                color: white;
                border: 1px solid #BDC3C7;
                border-radius: 6px;
                font-family: Microsoft YaHei;
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #2C3E50;
                color: white;
            }
        """)
        self.tabs.setStyleSheet("""
            QTabBar::tab {
                min-width: 120px;
                height: 32px;
                padding: 10px 24px;
                background: #1E272E;
                border: 1px solid #666;
                border-radius: 6px;
                font-family: Microsoft YaHei;
                font-size: 13px;
                color: white;
            }
            QTabBar::tab:selected {
                background: #3498DB;
                color: white;
                font-weight: bold;
            }
            QTabWidget::pane {
                border-top: 2px solid #3498DB;
                background: #2C3E50;
            }
        """)
        self.theme_btn.setStyleSheet("""
            QPushButton {
                background-color: #2C3E50;
                color: white;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #1E272E;
            }
        """)
        # 对子页面统一设置
        self.processing_widget.setStyleSheet("background-color: #2C3E50; color: white;")
        self.plot_widget.setStyleSheet("background-color: #2C3E50;")
        self.table_widget.setStyleSheet("""
            QTableWidget {
                background-color: #34495E;
                color: white;
            }
            QHeaderView::section {
                background: #2C3E50;
                color: white;
            }
            QTableWidget::item:selected {
                background-color: #2980B9;
            }
        """)

    def apply_light_theme(self):
        self.setStyleSheet("background-color: #F5F5F5; color: black;")
        self.findChild(QLabel).setStyleSheet("""
            QLabel {
                color: #2C3E50;
                margin-bottom: 20px;
                border-bottom: 2px solid #3498DB;
                padding-bottom: 10px;
            }
        """)
        self.processing_widget.metrics_table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                color: black;
                border: 1px solid #BDC3C7;
                border-radius: 6px;
                font-family: Microsoft YaHei;
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #3498DB;
                color: white;
            }
        """)
        self.tabs.setStyleSheet("""
            QTabBar::tab {
                min-width: 120px;
                height: 32px;
                padding: 10px 24px;
                background: #ECF0F1;
                border: 1px solid #BDC3C7;
                border-radius: 6px;
                font-family: Microsoft YaHei;
                font-size: 13px;
            }
            QTabBar::tab:selected {
                background: #3498DB;
                color: white;
                font-weight: bold;
            }
            QTabWidget::pane {
                border-top: 2px solid #3498DB;
                background: #FFFFFF;
            }
        """)
        self.theme_btn.setStyleSheet("""
            QPushButton {
                background-color: #34495E;
                color: white;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #2C3E50;
            }
        """)
        self.processing_widget.setStyleSheet("")
        self.plot_widget.setStyleSheet("")
        self.table_widget.setStyleSheet("")

    def calculate_score(self, metrics):
        """计算加权综合评分"""
        return sum(metrics.get(k, 0) * v for k, v in WEIGHTS.items())

    def create_table_widget(self, initial_results, new_results):
        """创建综合评分对比表格"""
        table = QTableWidget()
        table.setRowCount(len(initial_results))
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["图像名称", "原始评分", "增强后评分"])
        table.verticalHeader().setVisible(False)

        # 样式优化
        table.setStyleSheet("""
            QTableWidget {
                background: white;
                border: 1px solid #BDC3C7;
                border-radius: 6px;
                font-family: Microsoft YaHei;
            }
            QHeaderView::section {
                background: #3498DB;
                color: white;
                padding: 8px;
                border: none;
            }
            QTableWidget::item:hover {
                background-color: #ECF0F1;
            }
            QTableWidget::item:selected {
                background-color: #3498DB;
                color: white;
            }
        """)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # 填充数据
        for row, (img_name, init_metrics) in enumerate(initial_results):
            new_metrics = next((m for i, m in new_results if i == img_name), {})

            init_score = self.calculate_score(init_metrics)
            new_score = self.calculate_score(new_metrics)

            # 插入数据
            table.setItem(row, 0, QTableWidgetItem(img_name))
            table.setItem(row, 1, QTableWidgetItem(f"{init_score:.4f}"))
            table.setItem(row, 2, QTableWidgetItem(f"{new_score:.4f}"))

            # 设置标注
            if new_score > init_score:
                color = QColor(46, 204, 113)  # 绿色
                arrow = "↑"
            elif new_score < init_score:
                color = QColor(231, 76, 60)  # 红色
                arrow = "↓"
            else:
                color = QColor(241, 196, 15)  # 黄色
                arrow = "→"

            table.setItem(row, 2, QTableWidgetItem(f"{new_score:.4f}  {arrow}"))
            table.item(row, 2).setBackground(QBrush(color))
            table.item(row, 2).setForeground(QBrush(Qt.white))

        return table

    def create_plot_widget(self, initial_results, new_results):
        """创建评分趋势折线图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.subplots_adjust(bottom=0.25)

        # 计算所有评分
        init_scores = [self.calculate_score(m) for _, m in initial_results]
        new_scores = [self.calculate_score(next((m for i, m in new_results if i == img), {}))
                      for img, _ in initial_results]
        images = [img for img, _ in initial_results]

        # 绘制折线
        x = range(len(images))
        ax.plot(x, init_scores, marker='o', color='#3498DB', linewidth=2, label='原始评分')
        ax.plot(x, new_scores, marker='s', color='#E74C3C', linestyle='--', linewidth=2, label='增强评分')

        # 样式优化
        ax.set_xticks(x)
        ax.set_xticklabels(images, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel("综合评分", fontsize=12)
        ax.set_title("综合评分趋势对比", fontsize=14, pad=20)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#F5F5F5')
        ax.tick_params(axis='x', rotation=30, labelsize=10)
        for label in ax.get_xticklabels():
            label.set_fontstyle('italic')

        # 图例美化
        legend = ax.legend(loc='upper right', fontsize=10, frameon=True)
        legend.get_frame().set_edgecolor('#BDC3C7')
        legend.get_frame().set_linewidth(1)
        legend.get_frame().set_facecolor('#FFFFFF')

        canvas = FigureCanvas(fig)
        canvas.setStyleSheet("background: white; border-radius: 8px;")
        return canvas

    def create_imageProcessing_widget(self, initial_results=None, new_results=None):
        """创建在线图片处理系统，并放入滚动区域"""
        self.processing_widget = ImageProcessingWidget()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.processing_widget)
        return scroll_area


# 以下函数保持不变（show_metrics_gui, load_json_file, main）

def show_metrics_gui(initial_results, new_results):
    app = QApplication(sys.argv)
    window = MetricsWindow(initial_results, new_results)
    # window.setWindowOpacity(0.0)
    window.show()
    # for i in range(1, 11):
    #    QtCore.QTimer.singleShot(i * 40, lambda op=i: window.setWindowOpacity(op / 10))
    sys.exit(app.exec_())


def load_json_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: 文件 '{file_path}' 未找到。")
        sys.exit(1)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: 文件 '{file_path}' 格式错误。")
        sys.exit(1)


def main():
    initial_results = load_json_file(pyqt_initial_files)
    new_results = load_json_file(pyqt_files)
    show_metrics_gui(initial_results, new_results)


if __name__ == '__main__':
    main()