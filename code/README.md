# LogNormMix-Net Temporal Point Process

The TPP code is adapted from the Pytorch implementation of the paper ["Intensity-Free Learning of Temporal Point Processes"](https://openreview.net/forum?id=HygOjhEYDH), Oleksandr Shchur, Marin Biloš and Stephan Günnemann, ICLR 2020.

## Using your own data
You can save your custom dataset in the format used in our code as follows:

```python
dataset = {
    "sequences": [
        {"arrival_times": [0.2, 4.5, 9.1], "src_marks": [1, 0, 4], "dst_marks": [10, 2, 19], "meta": [1, 1, 2], "t_start": 0.0, "t_end": 10.0},
        {"arrival_times": [2.3, 3.3, 5.5], "src_marks": [4, 3, 2], "dst_marks": [14, 12, 1], "meta": [0, 1, 1], "t_start": 0.0, "t_end": 10.0},
    ],
    "num_src_marks": 5,
    "num_dst_marks": 20,
    "num_meta_classes": 3,
}
torch.save(dataset, "data/my_dataset.pkl")
```
