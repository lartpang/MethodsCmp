## UC-Net

```shell
$ python main.py --gpu 1 --method-names ucnet
 ==>> PROCESSING THE METHOD ucnet... <<== 
Counting Number of Ops. & Params. for Generator
[ucnet] FLOPs: 32.281G, Params: 31.263M
Counting GPU memory for Generator
[ucnet] GPU 1 MEM: Total: 1231.250MB | Model: 268.000MB | Other: 963.250MB
Counting FPS for Generator with 100 data on gpu
[ucnet] FPS: 75.570
```

## Reference

- <http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_UC-Net_Uncertainty_Inspired_RGB-D_Saliency_Detection_via_Conditional_Variational_Autoencoders_CVPR_2020_paper.pdf>
- <https://github.com/JingZhang617/UCNet>

```
@inproceedings{Zhang2020UCNet,
  title={UC-Net: Uncertainty Inspired RGB-D Saliency Detection via Conditional Variational Autoencoders},
  author={Zhang, Jing and Fan, Deng-Ping and Dai, Yuchao and Anwar, Saeed and Sadat Saleh, Fatemeh and Zhang, Tong and Barnes, Nick},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2020}
}

@article{zhang2020uncertainty,
  title={Uncertainty Inspired RGB-D Saliency Detection},
  author={Jing Zhang and Deng-Ping Fan and Yuchao Dai and Saeed Anwar and Fatemeh Saleh and Sadegh Aliakbarian and Nick Barnes},
  journal={arXiv preprint arXiv:2009.03075},
  year={2020}
}
```
