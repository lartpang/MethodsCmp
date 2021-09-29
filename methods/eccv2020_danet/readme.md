## DANet

```
$ python main.py --gpu 1 --method-names danet
 ==>> PROCESSING THE METHOD danet... <<== 
Counting Number of Ops. & Params. for DANet_V19
[danet] FLOPs: 157.757G, Params: 31.991M
Counting GPU memory for DANet_V19
[danet] GPU 1 MEM: Total: 1401.250MB | Model: 436.000MB | Other: 965.250MB
Counting FPS for DANet_V19 with 100 data on gpu
[danet] FPS: 59.050
```

## Reference

- <https://arxiv.org/pdf/2007.06811.pdf>
- <https://github.com/Xiaoqi-Zhao-DLUT/DANet-RGBD-Saliency>

```
@inproceedings{DANet,
  title={A Single Stream Network for Robust and Real-time RGB-D Salient Object Detection},
  author={Zhao, Xiaoqi and Zhang, Lihe and Pang, Youwei and Lu, Huchuan and Zhang, Lei},
  booktitle=ECCV,
  year={2020}
}
```
