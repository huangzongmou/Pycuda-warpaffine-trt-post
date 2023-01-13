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

mod = SourceModule("""
    __device__ void affine_project(float* matrix, int x, int y, float* proj_x, float* proj_y){

        // matrix
        // m0, m1, m2
        // m3, m4, m5
        *proj_x = matrix[0] * x + matrix[1] * y + matrix[2];
        *proj_y = matrix[3] * x + matrix[4] * y + matrix[5];
    }
    __global__ void warp_affine_bilinear_kernel(
    unsigned char *src, int *src_info, float *dst, int *dst_info,
    unsigned char *fill_value, float *affine)
    {
        int src_line_size = src_info[0];
        int src_width     = src_info[1];
        int src_height    = src_info[2];
        int dst_line_size = dst_info[0];
        int dst_width     = dst_info[1];
        int dst_height    = dst_info[2];
        
        int dx = blockDim.x * blockIdx.x + threadIdx.x; 
        int dy = blockDim.y * blockIdx.y + threadIdx.y;
        if (dx >= dst_width || dy >= dst_height)  return;



        float c0 = fill_value[0], c1 = fill_value[1], c2 = fill_value[2];
        float src_x = 0; float src_y = 0;

        affine_project(affine, dx, dy, &src_x, &src_y);

        if(src_x < -1 || src_x >= src_width || src_y < -1 || src_y >= src_height){
        // out of range
        // src_x < -1时，其高位high_x < 0，超出范围
        // src_x >= -1时，其高位high_x >= 0，存在取值
        }else{
            int y_low = floorf(src_y);
            int x_low = floorf(src_x);
            int y_high = y_low + 1;
            int x_high = x_low + 1;


            unsigned char const_values[] = {fill_value[0], fill_value[1], fill_value[2]};
            float ly    = src_y - y_low;
            float lx    = src_x - x_low;
            float hy    = 1 - ly;
            float hx    = 1 - lx;
            float w1    = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
            unsigned char* v1 = const_values;
            unsigned char* v2 = const_values;
            unsigned char* v3 = const_values;
            unsigned char* v4 = const_values;

            if(y_low >= 0){
                if (x_low >= 0)
                    v1 = src + y_low * src_line_size + x_low * 3;

                if (x_high < src_width)
                    v2 = src + y_low * src_line_size + x_high * 3;
            }
            
            if(y_high < src_height){
                if (x_low >= 0)
                    v3 = src + y_high * src_line_size + x_low * 3;

                if (x_high < src_width)
                    v4 = src + y_high * src_line_size + x_high * 3;
            }
            
            c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
            c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
            c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
        }

        //unsigned char * pdst = dst + dy * dst_line_size + dx * 3;
        //[0] = c0; pdst[1] = c1; pdst[2] = c2;

        float* pdst = dst + dy * dst_line_size + dx * 3;
        pdst[0] = c0/255.0; pdst[1] = c1/255.0; pdst[2] = c2/255.0;

}
""")

def affine_compute(src_size, dst_size):
    i2d = np.zeros([3,3])
    scale_x = dst_size[0] / src_size[0]
    scale_y = dst_size[1] / src_size[1]
    scale = min(scale_x, scale_y)

    i2d[0][0] = scale
    i2d[0][1] = 0
    i2d[0][2] = -scale * src_size[0]  * 0.5  + dst_size[0] * 0.5 + scale * 0.5 - 0.5
    i2d[1][0] = 0
    i2d[1][1] = scale
    i2d[1][2] = -scale * src_size[1] * 0.5 + dst_size[1] * 0.5 + scale * 0.5 - 0.5
    i2d[2][2] = 1
    d2i = np.linalg.inv(i2d)

    return d2i.astype(np.float32)

class Warpaffine(object):
    def __init__(self, dst_size=(640,640),src_size=(1920,1080)):
        super(Warpaffine, self).__init__()
        self.src_size = src_size
        self.dst_size = dst_size
        self.block_size = (32, 32,1)
        self.grid_size = ((dst_size[0] + 31) // 32, (dst_size[1] + 31) // 32, 1)
        
        self.pdst_host = cuda.register_host_memory(np.ones((dst_size[0],dst_size[1],3)).astype(np.float32))
        self.pdst_device = cuda.mem_alloc(self.pdst_host.nbytes)

        self.img_host = cuda.register_host_memory(np.ones((src_size[1],src_size[0],3)).astype(np.uint8))
        self.img_device = cuda.mem_alloc(src_size[0]*src_size[1]*3)
 
        self.src_info = np.array([src_size[0]*3,src_size[0],src_size[1]]).astype(np.int32)
        self.dst_info = np.array([dst_size[0]*3,dst_size[0],dst_size[1]]).astype(np.int32)
        self.fill_value = np.array([114,114,114]).astype(np.int8)
        self.affine = affine_compute(src_size,dst_size)
        
        self.stream = cuda.Stream()
        self.func = mod.get_function("warp_affine_bilinear_kernel")

    def up_information(self,img):
        self.src_size = (img.shape[1],img.shape[0])
        self.img_device = cuda.mem_alloc(img.nbytes)

        self.img_host = cuda.register_host_memory(np.ones((self.src_size[1],self.src_size[0],3)).astype(np.uint8))
        self.src_info = np.array([self.src_size[0]*3,self.src_size[0],self.src_size[1]]).astype(np.int32)
        self.affine = affine_compute(self.src_size,self.dst_size)


    def preprocess(self,img):

        if operator.eq(self.src_size,(img.shape[1],img.shape[0])) is False:
            self.up_information(img)
            print("1111")
        
        
        t1 = time.time()

        np.copyto(self.img_host,img.data)

        t2 = time.time()
        cuda.memcpy_htod_async(self.img_device, self.img_host, self.stream)
        self.func(self.img_device,cuda.In(self.src_info),self.pdst_device,cuda.In(self.dst_info),\
            cuda.In(self.fill_value),cuda.In(self.affine),stream=self.stream,block=self.block_size,grid=self.grid_size)
        t3 = time.time()
        cuda.memcpy_dtoh_async(self.pdst_host, self.pdst_device, self.stream)
        t4 = time.time()
        self.stream.synchronize()

        t5 = time.time()
        
        print("t2:%f,t3:%f,t4:%f,t5:%f"%(t2-t1,t3-t2,t4-t3,t5-t1))

        return self.pdst_host


if __name__ == "__main__":

    img =cv2.imread("dog1.jpg")
    warpaffine1 = Warpaffine(dst_size=(640,640))
    print(img.shape)
    for _ in range(10000):
        # img =cv2.imread("cat1.png")
        # pdst_img = warpaffine1.preprocess(img)
        # img =cv2.imread("dog1.jpg")
        
        pdst_img = warpaffine1.preprocess(img)
        # time.sleep(0.001)
        # pdst_img = warpaffine(img,(640,640))
    # print(pdst_img[:,0,0])
    # img =cv2.imwrite("my.jpg",pdst_img*255)


