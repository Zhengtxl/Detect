import os
import shutil
 
import cv2

# 获得当前路径
dir = r"E:\Python\yolo11\datasets\1\\"
os.chdir(dir)
target_dir = dir + "images"
label_dir = dir + "labels"
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
        source_image = cv2.imread(source_file_path)
        source_image=cv2.resize(source_image,(640,640))
        target_file_path = os.path.join(target_dir, filename)
        # 复制文件到目标目录
        cv2.imwrite(target_file_path,source_image)
        label_name = label_dir + '\\' + (filename.split(' ')[0]).split('.')[0] + '.txt'
        open(label_name, 'w').close()
# 将不包含掩码图的ng照片放入img中，同时将有缺陷部分记录下来
mark_txt = ng_dir + "\\mark.txt"

with open(mark_txt, encoding='utf-8') as file:
    lines = ''.join(file.readlines()).splitlines()
    for line in lines[:-5]:
        # 获取路径
        defect_name = ng_dir + '\\' + line.split(' ')[0]
        mask_name = ng_dir + '\\' + line.split(' ')[1]
        defect_image = cv2.imread(defect_name)
        mask_image = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        defect_image = cv2.resize(defect_image, (640, 640))
        mask_image = cv2.resize(mask_image, (640, 640))

        img_x, img_y = defect_image.shape[0], defect_image.shape[1]
        print(img_x, img_y, defect_name)
        # 使用反转后的掩膜图提  取缺陷区域
        defect_area = cv2.bitwise_and(defect_image, defect_image, mask=mask_image)
        # 查找掩膜图中的轮廓
        contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        target_dir_path = os.path.join(target_dir, line.split(' ')[0])
        label_name = label_dir + '\\' + (line.split(' ')[0]).split('.')[0] + '.txt'
        cv2.imwrite(target_dir_path, defect_image)
        # 178.99968 384.99968 15539.00032 31744.99968
        with open(label_name, 'w') as file1:
            for contour in contours:
                # 计算轮廓的外接矩形
                x, y, w, h = cv2.boundingRect(contour)
                x_ratio = x / img_y
                # 计算相对于图像尺寸的比例
                y_ratio = y / img_x
                w_ratio = w/ img_x
                h_ratio = h / img_y
                x_center=(x+w//2)/img_x
                y_center=(y+h//2)/img_y
                # cv2.rectangle(defect_image,(x,y),(x+w,y+h),(0,255,0),2)
                # cv2.imshow('Defect Detection',defect_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # 将信息格式化为字符串并写入文件
                file1.write(f"0 {x_center:.6f} {y_center:.6f} {w_ratio:.6f} {h_ratio:.6f}\n")
