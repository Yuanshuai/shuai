import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
from app import Chart_conversion
from tkinter import simpledialog
def on_drop(event):
    # 获取拖拽的文件路径
    file_path = root.tk.splitlist(event.data)
    # 弹出对话框让用户输入Excel文件名
    user_input = tk.simpledialog.askstring("输出文件", "Excel表格的文件名：", parent=root)
    if user_input:  # 检查用户是否输入了文件名

        # 创建Chart_conversion对象并处理文件
        ee = Chart_conversion(file_path[0], f"{user_input}.xlsx",merge_cells_var.get())
        ee.do()
    else:
        file_path_label.config(text="未输入文件名")

# 创建主窗口
root = TkinterDnD.Tk()
root.title("图表转化器")
root.geometry("400x200")  # 设置窗口大小

# 使窗口支持文件拖拽
root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', on_drop)

# 创建一个标签显示选中的文件路径
file_path_label = tk.Label(root, text="拖拽图片到这里", fg="gray", font=("Arial", 12))
file_path_label.pack(pady=20)  # 垂直外边距

# 添加复选框，用于选择是否开启合并单元格检测
merge_cells_var = tk.BooleanVar()
merge_cells_checkbox = tk.Checkbutton(root, text="是否开启合并单元格检测", variable=merge_cells_var)
merge_cells_checkbox.pack(pady=10)

# 运行主循环
root.mainloop()