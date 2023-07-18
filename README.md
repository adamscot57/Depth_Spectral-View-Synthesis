# Depth_Spectral-View-Synthesis

An unofficial PyTorch implementation of Self-supervised Depth Estimation from Spectral Consistency and Novel View Synthesis, 2022 IJCNN



# Spectral consistency depth estimation
**Train:**
```
python train.py --data_path kitti_data --log_dir logs/  --model_name stereo_depth_clue   --frame_ids 0  --use_stereo  --scheduler_step_size 5 --split eigen_full --disparity_smoothness 0
```

**Infer:**
```
python test_simple.py  --image_path test_img/  --model_path logs/stereo_depth_clue/models/weights_19  --num_layers 18
```


# Stereo synthesis depth estimation
**Train:**

```
python stereo-synthesis/train.py

Training and test images should be put in data/target/train:
Input: input left
Synthesized_image: syn right
Real_image: real right
```

**Infer:**
```
CUDA_LAUNCH_BLOCKING=1 python stereo-synthesis/infer.py
```

# Uncertainty for depth fusion
```
python mono_uncertainty/generate_maps.py --data_path kitti_data  --load_weights_folder S/S/Monodepth2-Post/models/weights_19/ \
                        --post_process \
                        --eval_split eigen_benchmark \
                        --eval_mono \
                        --output_dir uncertainty/post/ \
```

```
python mono_uncertainty/generate_maps.py --data_path kitti_data  --load_weights_folder S/S/Monodepth2-Post/models/weights_19/ \
                        --post_process \
                        --eval_split eigen_benchmark \
                        --eval_mono \
                        --output_dir uncertainty/post/ \
```

```
python merge_fusion.py
```


## Cite
Implementation is based on the method in:
```
@inproceedings{lu2022self,
  title={Self-supervised Depth Estimation from Spectral Consistency and Novel View Synthesis},
  author={Lu, Yawen and Lu, Guoyu},
  booktitle={2022 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2022},
  organization={IEEE}
}
```

