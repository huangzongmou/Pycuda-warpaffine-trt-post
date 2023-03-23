# 基于pycuda的yolo前后处理

使用pycuda对yolo前后处理进gpu加速。

### 使用例子
    inference = TRT_inference("./weights/yolov5s.onnx")
    img = cv2.imread("dog1.jpg")
    boxs = inference(img)
    for box in boxs:
        cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(255,0,0),2)
    cv2.imwrite("test.jpg",img)