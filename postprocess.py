import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

import numpy as np
import torch
import cv2
import torchvision
import time
import operator
from warpaffine import Warpaffine



class gpu_decode(object):
    def __init__(self, rows, cols, confidence_threshold = 0.6,nms_threshold = 0.45,stream=None):
        super(gpu_decode, self).__init__()
        self.rows = rows
        self.cols = cols

        self.block = 512 if rows > 512 else rows
        self.grid = (rows + self.block - 1) // self.block
        self.block = (self.block,1,1)
        self.grid = (self.grid,1,1)

        self.max_objects = 1000
        self.NUM_BOX_ELEMENT = 7
        self.num_bboxes = cuda.In(np.array([rows]).astype(np.int32))
        self.num_classes = cuda.In(np.array([cols-5]).astype(np.int32))
        self.confidence_threshold = cuda.In(np.array([confidence_threshold]).astype(np.float32))
        self.nms_threshold = cuda.In(np.array([nms_threshold]).astype(np.float32))

        self.nms_block = 512 if self.max_objects > 512 else self.max_objects
        self.nms_grid = (self.max_objects + self.nms_block - 1) / self.nms_block;
        self.nms_block = (self.nms_block,1,1)
        self.nms_grid = (self.nms_grid,1,1)

        if stream == None:
            self.stream = cuda.Stream()
        else:
            self.stream = stream

        # self.predict_host = cuda.register_host_memory(np.ones((1,self.rows,self.cols)).astype(np.float32))
        # self.predict_device = cuda.mem_alloc(self.predict_host.nbytes)

        self.output_host = cuda.register_host_memory(np.ones((self.max_objects, self.NUM_BOX_ELEMENT)).astype(np.float32))
        self.output_device_nbytes = self.output_host.nbytes
        self.output_device = cuda.mem_alloc(self.output_device_nbytes)
        self.max_objects = cuda.In(np.array([self.max_objects]).astype(np.int32))
        self.NUM_BOX_ELEMENT = cuda.In(np.array([self.NUM_BOX_ELEMENT]).astype(np.int32))

        self.filter_boxs = cuda.InOut(np.array([0]).astype(np.int32))  #获取第一次过滤后的box数量

        self.decode_kernel,self.fast_nms_kernel = self.cuda_func()

    def cuda_func(self):
        mod = SourceModule("""
        __device__ void affine_project(float* matrix, float x, float y, float* ox, float* oy){
        *ox = matrix[0] * x + matrix[1] * y + matrix[2];
        *oy = matrix[3] * x + matrix[4] * y + matrix[5];
        }

        __global__ void decode_kernel(
            float* predict, int* num_bboxes, int* num_classes, float* confidence_threshold, 
            float* invert_affine_matrix, float* parray, int* max_objects, int* filter_boxs, int* NUM_BOX_ELEMENT
        )
        {  
            int position = blockDim.x * blockIdx.x + threadIdx.x;
            if (position >= *num_bboxes) return;

            float* pitem     = predict + (5 + *num_classes) * position;
            float objectness = pitem[4];
            if(objectness < *confidence_threshold)
                return;

            float* class_confidence = pitem + 5;
            float confidence        = *class_confidence++;
            int label               = 0;
            for(int i = 1; i < *num_classes; ++i, ++class_confidence){
                if(*class_confidence > confidence){
                    confidence = *class_confidence;
                    label      = i;
                }
            }

            confidence *= objectness;
            if(confidence < *confidence_threshold)
                return;

            int index = atomicAdd(filter_boxs, 1);
            if(index >= *max_objects)
                return;

            float cx         = *pitem++;
            float cy         = *pitem++;
            float width      = *pitem++;
            float height     = *pitem++;
            float left   = cx - width * 0.5f;
            float top    = cy - height * 0.5f;
            float right  = cx + width * 0.5f;
            float bottom = cy + height * 0.5f;

            affine_project(invert_affine_matrix, left,  top,    &left,  &top);
            affine_project(invert_affine_matrix, right, bottom, &right, &bottom);
            // left, top, right, bottom, confidence, class, keepflag
            float* pout_item = parray + index * (*NUM_BOX_ELEMENT);
            *pout_item++ = left;
            *pout_item++ = top;
            *pout_item++ = right;
            *pout_item++ = bottom;
            *pout_item++ = confidence;
            *pout_item++ = label;
            *pout_item++ = 1; // 1 = keep, 0 = ignore
        }

        __device__ float box_iou(
            float aleft, float atop, float aright, float abottom, 
            float bleft, float btop, float bright, float bbottom
        ){

            float cleft 	= max(aleft, bleft);
            float ctop 		= max(atop, btop);
            float cright 	= min(aright, bright);
            float cbottom 	= min(abottom, bbottom);
            
            float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
            if(c_area == 0.0f)
                return 0.0f;
            
            float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
            float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
            return c_area / (a_area + b_area - c_area);
        }

        __global__ void fast_nms_kernel(float* bboxes, int*filter_boxs, int* max_objects, float* threshold, int* NUM_BOX_ELEMENT){

            int position = (blockDim.x * blockIdx.x + threadIdx.x);
            int count = min(*filter_boxs, *max_objects);

            
            if (position >= count) 
                return;
            
            // left, top, right, bottom, confidence, class, keepflag
            float* pcurrent = bboxes + position * (*NUM_BOX_ELEMENT);
            for(int i = 0; i < count; ++i){
                float* pitem = bboxes + i * (*NUM_BOX_ELEMENT);
                if(i == position || pcurrent[5] != pitem[5]) continue;

                if(pitem[4] >= pcurrent[4]){
                    if(pitem[4] == pcurrent[4] && i < position)
                        continue;

                    float iou = box_iou(
                        pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                        pitem[0],    pitem[1],    pitem[2],    pitem[3]
                    );

                    if(iou > *threshold){
                        pcurrent[6] = 0;  // 1=keep, 0=ignore
                        return;
                    }
                }
            }
        } 

        """)
        return mod.get_function("decode_kernel"),mod.get_function("fast_nms_kernel")
    

    def decode_kernel_invoker(self,predict, affine):
        
        # np.copyto(self.predict_host,predict[0].data)

        # cuda.memcpy_htod_async(self.predict_device, self.predict_host, self.stream)

        self.decode_kernel(predict,\
            self.num_bboxes,\
            self.num_classes,\
            self.confidence_threshold,\
            cuda.In(affine),\
            self.output_device,\
            self.max_objects,\
            self.filter_boxs,\
            self.NUM_BOX_ELEMENT,\
            stream=self.stream,block=self.block,grid=self.grid)

        # self.stream.synchronize()
        # cuda.memcpy_dtoh_async(self.output_host, self.output_device, self.stream)
        # print(len(self.output_host[self.output_host[:,6]>0]))
        # print("111111111")
        self.fast_nms_kernel(self.output_device,self.filter_boxs,self.max_objects,self.nms_threshold,self.NUM_BOX_ELEMENT,stream=self.stream,block=self.block,grid=self.grid)
        cuda.memcpy_dtoh_async(self.output_host, self.output_device, self.stream)
        self.stream.synchronize()
        
        cuda.memset_d8(self.output_device, 0, self.output_device_nbytes)  #清空
        cuda.memset_d8(self.filter_boxs, 0, sizeof(int32)) #清空计数

        return self.output_host[self.output_host[:,6]>0]

    def __call__(self, predict, affine):
        return self.decode_kernel_invoker(predict, affine)





if __name__ == "__main__":
    device = "cpu"
    model = torch.jit.load("yolov5s.torchscript")  
    warpaffine = Warpaffine(dst_size=(640,640))
    postprocess = gpu_decode(rows=25200, cols=85)
    img = cv2.imread("bus.jpg")
    pdst_img = warpaffine(img)
    pdst_img = torch.from_numpy(pdst_img).to(device)
    predict = model(pdst_img)[0].numpy()

    img1 = warpaffine(img)*255
    img1 = img1[0].transpose(1, 2, 0)
    print(img1.shape)
    img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)

    t1 = time.time()
    for _ in range(1000):
        boxs = postprocess(predict)
    t2 = time.time()
    print(t2-t1)
    for box in boxs:
        cv2.rectangle(img1,(box[0],box[1]),(box[2],box[3]),(255,0,0),2)
    cv2.imwrite("test.jpg",img1)
   

