## HDFNet

```shell
$ python main.py --gpu 1 --method-names hdfnet_res50 hdfnet_vgg16
 ==>> PROCESSING THE METHOD hdfnet_res50... <<== 
Counting Number of Ops. & Params. for HDFNet_Res50
[hdfnet_res50] FLOPs: 89.314G, Params: 153.192M
Counting GPU memory for HDFNet_Res50
[hdfnet_res50] GPU 1 MEM: Total: 2755.250MB | Model: 1790.000MB | Other: 965.250MB
Counting FPS for HDFNet_Res50 with 100 data on gpu
[hdfnet_res50] FPS: 49.590                                                                                                                                                       
 ==>> PROCESSING THE METHOD hdfnet_vgg16... <<== 
Counting Number of Ops. & Params. for HDFNet_VGG16
[hdfnet_vgg16] FLOPs: 183.218G, Params: 44.148M
Counting GPU memory for HDFNet_VGG16
[hdfnet_vgg16] GPU 1 MEM: Total: 1349.250MB | Model: 318.000MB | Other: 1031.250MB
Counting FPS for HDFNet_VGG16 with 100 data on gpu
[hdfnet_vgg16] FPS: 53.107
```
## Reference

- <https://arxiv.org/pdf/2007.06227.pdf>
- <https://github.com/lartpang/HDFNet>

```
@inproceedings{HDFNet-ECCV2020,
    author = {Youwei Pang and Lihe Zhang and Xiaoqi Zhao and Huchuan Lu},
    title = {Hierarchical Dynamic Filtering Network for RGB-D Salient Object Detection},
    booktitle = ECCV,
    year = {2020}
}
```
