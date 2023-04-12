import tensorrt as trt
import cv2
from warpaffine import Warpaffine
from postprocess import gpu_decode
# import pycuda.autoinit  #负责数据初始化，内存管理，销毁等
import pycuda.driver as cuda  #GPU CPU之间的数据传输
import time
import os
import copy

class TRT_inference(object):
    def __init__(self, model_path,model="yolov5"):
        super(TRT_inference, self).__init__()

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 <<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20) # 1 MiB
        # config.set_flag(trt.BuilderFlag.INT8)

        engine_path = model_path.replace("onnx","engine")
        if os.path.isfile(engine_path):
            with open(engine_path, "rb") as f:
                serialized_engine = f.read()
        else:
            parser = trt.OnnxParser(network, logger)
            success = parser.parse_from_file(model_path)
            if not success:
                print("Error handling code here")

            serialized_engine = builder.build_serialized_network(network, config)
            
            with open(engine_path, "wb") as f:
                f.write(serialized_engine)

        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = engine.create_execution_context()

        dst_size = self.context.get_binding_shape(engine["images"])[2:][::-1]   #(w,h)
        rows,cols = self.context.get_binding_shape(engine["output"])[1:]
        self.d_output = cuda.mem_alloc(trt.volume(self.context.get_binding_shape(engine["output"]))*4)

        self.stream = cuda.Stream()
        self.warpaffine = Warpaffine(dst_size=dst_size,stream=self.stream)
        self.postprocess = gpu_decode(rows=rows, cols=cols,model=model, stream=self.stream)

    def __call__(self, img):
        pdst_img,affine = self.warpaffine(img)
        self.context.execute_async_v2(bindings=[int(pdst_img), int(self.d_output)], stream_handle=self.stream.handle)
        boxs = self.postprocess(self.d_output,affine)
        return boxs

if __name__ == "__main__":

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
