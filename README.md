## Multi-degradation-adaptation network for fundus image enhancement with degradation representation learning (Medical Image Analysis 2024)
---
Official Pytorch implementation
### Training
---
```sh train_from_scratch.sh```
### Testing
---
The training code automatically saves weights in the folder ```./myoutput``` and indexes each training. To select a specific network weight for testing, replace ```XX``` and ```YYY``` with the corresponding index and epoch numbers, respectively.  
```
python eval.py --framework clf --load_index XX --load_epoch YYY --dataset DATASETNAME dataset_path DATASETPATH --output_path OUTPUTPATH
```
### Citations
---
If you find our code useful, please consider citing our paper.

```
@article{MDA-Net,
          title = {Multi-degradation-adaptation network for fundus image enhancement with degradation representation learning},
          journal = {Medical Image Analysis},
          volume = {97},
          pages = {103273},
          year = {2024},
          author = {Ruoyu Guo and Yiwen Xu and Anthony Tompkins and Maurice Pagnucco and Yang Song}
}
```

