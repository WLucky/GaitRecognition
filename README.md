# CV Homework: Gait Recognition

## Experiment Results
Test accuracy of carry bag(**BG**),wear clothes(**CL**),normal(**NM**)

![](https://github.com/WLucky/GaitRecognition/blob/main/Figs/test_acc.png)

## Train

**Pretreatment :**

```
python datasets/pretreatment.py \
	--input_path data/train \
	--output_path data/train_pkl
```

**Train :**

```
python -u main.py \
	--dataset_root ./data/train_pkl \
 	--log_to_file
```

## Inference

**Pretreatment :**

```
python inference/infer_pretreatment.py \
	--input_path data/test \
	--output_path data/test_pkl
```

**Inference :**

```
python -u inference.py
```



## Reference

https://github.com/ShiqiYu/OpenGait

GaitPart: Temporal Part-based Model for Gait Recognition

