# 基于pycuda的yolo前后处理

使用pycuda对yolo前后处理进gpu加速。

## Warpaffine.py 仿射变换实现yolo模型前处理
### 使用例子
    warpaffine = Warpaffine(dst_size=(640,384)) #dst_size 目标尺寸 #src_size 选填，运行过程中会自动更新
    img =cv2.imread("dog1.jpg")
    pdst_img = warpaffine(img) #缩放和填充和/255
    均在核函数里完成    
    cv2.imwrite("my.jpg",pdst_img*255)