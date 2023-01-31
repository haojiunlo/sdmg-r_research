# Training SDMG-R in xfun
1. Install mmocr by following [install.md](https://github.com/open-mmlab/mmocr/blob/main/docs/install.md)
2. backup the raw model
```bash
~$ cd YOUR_PATH_TO_mmocr
mv mmocr/datasets/kie_dataset.py mmocr/datasets/kie_dataset_raw.py  
mv mmocr/models/kie/heads/sdmgr_head.py mmocr/models/kie/heads/sdmgr_head_raw.py
mv mmocr/models/kie/losses/sdmgr_loss.py mmocr/models/kie/losses/sdmgr_loss_raw.py
mv mmocr/models/kie/extractors/sdmgr.py mmocr/models/kie/extractors/sdmgr_raw.py
mv tools/kie_test_imgs.py tools/kie_test_imgs_raw.py
mv mmocr/core/visualize.py mmocr/core/visualize_raw.py
```

3. replace files
```bash
cp sdmg-r_research/src/kie_dataset.py mmocr/mmocr/datasets/kie_dataset.py  
cp sdmg-r_research/src/sdmgr_head.py mmocr/mmocr/models/kie/heads/sdmgr_head.py
cp sdmg-r_research/src/sdmgr_loss.py mmocr/mmocr/models/kie/losses/sdmgr_loss.py
cp sdmg-r_research/src/sdmgr.py mmocr/mmocr/models/kie/extractors/sdmgr.py 
cp sdmg-r_research/src/kie_test_imgs.py mmocr/tools/kie_test_imgs.py 
cp sdmg-r_research/src/visualize.py mmocr/mmocr/core/visualize.py
cp sdmg-r_research/src/sdmgr_unet16_60e_xfun.py mmocr/configs/kie/sdmgr/sdmgr_unet16_60e_xfun.py
```

4. run
download xfun dataset [here](https://github.com/doc-analysis/XFUND/releases/tag/v1.0)
You should specify your xfun dataset path in `configs/kie/sdmgr/sdmgr_unet16_60e_xfun.py`
```bash
mmocr$ ./tools/dist_train.sh configs/kie/sdmgr/sdmgr_unet16_60e_xfun.py work_dir 1  # train
```

5. test and visualize the results

```bash
sh tools/kie_test_imgs.sh  configs/kie/sdmgr/sdmgr_unet16_60e_xfun.py work_dir/latest.pth ${out_put_dir}
```
