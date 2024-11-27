import os
import shutil
import cv2
# 获得当前路径
dir = r"D:\\datas\\1\\"
os.chdir(dir)
target_dir = dir + "img"
label_dir=dir+"label"
# 正常数据集放一起
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
if not os.path.exists(label_dir):
    os.makedirs(label_dir)
lists = os.listdir(dir)
print(lists)
# 获取ok和ng的路径
ok_dir, ng_dir = dir + lists[0], dir + lists[1]

# 将ok数据集中的所有bmp照片放入img中
# for filename in os.listdir(ok_dir):
#     if filename.endswith('.bmp'):
#         source_file_path = os.path.join(ok_dir, filename)
#         target_file_path = os.path.join(target_dir, filename)
#         print(source_file_path, target_file_path)
#         # 复制文件到目标目录
#         shutil.copy2(source_file_path, target_file_path)
# 将不包含掩码图的ng照片放入img中，同时将有缺陷部分记录下来
mark_txt=ng_dir+"\\mark.txt"

with open(mark_txt,encoding='utf-8') as file:
    lines=''.join(file.readlines()).splitlines()
    for line in lines[:5]:
        # 获取路径
        defect_name=ng_dir+'\\'+line.split(' ')[0]
        mask_name=ng_dir+'\\'+line.split(' ')[1]
        defect_image = cv2.imread(defect_name)
        mask_image = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        img_h,img_w=defect_image.shape[0],defect_image.shape[1]
        # 使用反转后的掩膜图提取缺陷区域
        defect_area = cv2.bitwise_and(defect_image, defect_image, mask=mask_image)
        # 查找掩膜图中的轮廓
        contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 在原图上绘制矩形框
        for contour in contours:
            # 计算轮廓的外接矩形
            x, y, w, h = cv2.boundingRect(contour)
            print(x/img_w, y/img_h, w/img_w, h/img_h)
            # 在原图上绘制矩形框
            cv2.rectangle(defect_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

             # 写入label文本中
            cv2.namedWindow("Defect Detection", 0)
            cv2.resizeWindow('Defect Detection', 512, 512)
            cv2.imshow('Defect Detection', defect_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print()