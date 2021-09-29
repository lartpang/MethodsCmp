## JL-DCF

```shell
$ python main.py --gpu 1 --method-names jldcf
 ==>> PROCESSING THE METHOD jldcf... <<== 
Counting Number of Ops. & Params. for JL_DCF
[jldcf] FLOPs: 1.722T, Params: 143.517M
Counting GPU memory for JL_DCF
[jldcf] GPU 1 MEM: Total: 1945.250MB | Model: 982.000MB | Other: 963.250MB
Counting FPS for JL_DCF with 100 data on gpu
[jldcf] FPS: 15.740
```

## Reference

- <http://openaccess.thecvf.com/content_CVPR_2020/papers/Fu_JL-DCF_Joint_Learning_and_Densely-Cooperative_Fusion_Framework_for_RGB-D_Salient_CVPR_2020_paper.pdf>
- <https://github.com/jiangyao-scu/JL-DCF-pytorch>

```
@inproceedings{Fu2020JLDCF,
title={JL-DCF: Joint Learning and Densely-Cooperative Fusion Framework for RGB-D Salient Object Detection},
author={Fu, Keren and Fan, Deng-Ping and Ji, Ge-Peng and Zhao, Qijun},
booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
pages={3052--3062},
year={2020}
}
    
@article{Fu2021siamese,
title={Siamese Network for RGB-D Salient Object Detection and Beyond},
author={Fu, Keren and Fan, Deng-Ping and Ji, Ge-Peng and Zhao, Qijun and Shen, Jianbing and Zhu, Ce},
journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
year={2021}
}
```
