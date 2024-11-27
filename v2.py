import os
import shutil

import cv2

# 获得当前路径
dir = r"D:\\datas\\1\\"
os.chdir(dir)
target_dir = dir + "img"
label_dir = dir + "label"
# 正常数据集放一起
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
if not os.path.exists(label_dir):
    os.makedirs(label_dir)
lists = os.listdir(dir)

# 获取ok和ng的路径
ok_dir, ng_dir = dir + lists[0], dir + lists[1]

# 将ok数据集中的所有bmp照片放入img中
for filename in os.listdir(ok_dir):
    if filename.endswith('.bmp'):
        source_file_path = os.path.join(ok_dir, filename)
        target_file_path = os.path.join(target_dir, filename)
        # 复制文件到目标目录
        shutil.copy2(source_file_path, target_file_path)
# 将不包含掩码图的ng照片放入img中，同时将有缺陷部分记录下来
mark_txt = ng_dir + "\\mark.txt"

with open(mark_txt, encoding='utf-8') as file:
    lines = ''.join(file.readlines()).splitlines()
    for line in lines:
        # 获取缺陷图片和掩码图
        defect_name = ng_dir + '\\' + line.split(' ')[0]
        mask_name = ng_dir + '\\' + line.split(' ')[1]
        # 读取文件
        defect_image = cv2.imread(defect_name)
        mask_image = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        # 缺陷图片长和宽
        img_h, img_w = defect_image.shape[0], defect_image.shape[1]
        # 使用反转后的掩膜图提取缺陷区域
        defect_area = cv2.bitwise_and(defect_image, defect_image, mask=mask_image)
        # 查找掩膜图中的轮廓
        contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 将所有的缺陷图写入img中
        target_dir_path = os.path.join(target_dir, line.split(' ')[0])
        shutil.copy2(defect_name, target_dir_path)
        # 打开一个文件以写入（如果文件不存在则创建）(line.split(' ')[0]).split('.')[0]第一个split获得文件名，第二个去除文件名的后缀
        label_name = label_dir + '\\' + (line.split(' ')[0]).split('.')[0] + '.txt'
        with open(label_name, 'w') as file:
            for contour in contours:
                # 计算轮廓的外接矩形
                x, y, w, h = cv2.boundingRect(contour)
                # 计算相对于图像尺寸的比例
                x_ratio = x / img_w
                y_ratio = y / img_h
                w_ratio = w / img_w
                h_ratio = h / img_h
                # 将信息格式化为字符串并写入文件
                file.write(f"0 {x_ratio:.6f} {y_ratio:.6f} {w_ratio:.6f} {h_ratio:.6f}\n")
