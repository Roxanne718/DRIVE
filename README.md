# Vessel Extraction

## Dataset

[DRIVE: Digital Retinal Images for Vessel Extraction](https://drive.grand-challenge.org/)

## Methods

- [Matched filtering](https://blog.csdn.net/qq_40511157/article/details/102770108)
- [Morphology](https://blog.csdn.net/virus1175/article/details/107126348?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522162641902416780357223650%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=162641902416780357223650&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_v2~rank_v29-2-107126348.first_rank_v2_pc_rank_v29&utm_term=%E7%9C%BC%E5%BA%95%E8%A1%80%E7%AE%A1%E5%88%86%E5%89%B2+%E5%BD%A2%E6%80%81%E5%AD%A6&spm=1018.2226.3001.4187)

## Run

Please download dataset to dataset/ firstly.

```
conda env create -f environment.yml
conda activate img_seg

python main.py --method matched_filtering
python main.py --method morphology
```