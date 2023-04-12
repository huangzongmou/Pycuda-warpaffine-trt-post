[toc]
___
# 基于pycuda的yolo前后处理

    1、使用pycuda对yolo前后处理进gpu加速。前处理包含操作：缩放、补边、bgr->rgb,转换维度（1,640,640,3）->（1,3,640,640）,除255.后处理支持yolov5、yolov8.
    2、yolov8模型需要./tool/add_transpose_node.py 将官方onnx模型输出进行转换和重命名(output:1,8400,84)，方便处理。



### 使用例子
```
    yolov8_inference = TRT_inference("./weights/yolov8n_transpose.onnx",model="yolov8")
    img = cv2.imread("./img/bus.jpg")
    img1 = copy.deepcopy(img)

    boxs = yolov8_inference(img)
    for box in boxs:
        cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,0),2)
    cv2.imwrite("./img/yolov8_test.jpg",img)

    yolov5_inference = TRT_inference("./weights/yolov5s.onnx",model="yolov5")
    boxs = yolov5_inference(img1)
    for box in boxs:
        cv2.rectangle(img1,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,0),2)
    cv2.imwrite("./img/yolov5_test.jpg",img1)
```