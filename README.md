## Suggestive Annotation

### Instructions to Run:

1. Download the gland segmentation challenge dataset and extract it into folder `Warwick QU Dataset (Released 2016_07_08)`.
2. To change all images to same dimensions run `reshape.py`.
   It generates the new images in `reshaped_warwick` directory.
   Move the images you don't want to start training with, into `images_train_eval` and also the corresponding annotations into `segments_train_eval`.
3. Now to generate the csv files, run `gen_csv.py`
4. Now, to run 4 sessions of training over 4 bootstrapped datasets, run `train.py`
5. To get the list of images to add for active learning, run `eval.py`. Make sure the `eval_data` flag in `eval.py` is `train_eval`
6. After this, to get the list of images to add to the training set (move from `train_eval` to `train` folder) run `active_selection.py`

Again goto step 3, till satifactory results are not obtained.
