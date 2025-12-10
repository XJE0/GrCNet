# GrCNet: Granularity-aware Graph Attention Network for Knowledge Graph Completion  
 

## Dependencies
- [PyTorch] == 2.7.1
- [python] == 3.9 
- [cuda] >= 12.6 
## Dataset:

- We use UMLS、Kinship、FB15k-237 and WN18RR datasets for knowledge graph link prediction. 
- UMLS、Kinship、FB15k-237 and WN18RR are included in the `data` directory. 

## Model Training
```
python main.py --data ./data/你自己的数据集路径/ --output_folder ./checkpoints/你自己的数据集输出文件/ --similarity_method cosine

# UMLS
python main.py --data ./data/umls/ --output_folder ./checkpoints/umls/out/ --similarity_method cosine

# Kinship
python main.py --data ./data/kinship/ --output_folder ./checkpoints/kinship/out/ --similarity_method cosine

# WN18RR
python main.py --data ./data/WN18RR/ --output_folder ./checkpoints/WN18RR/out/ --similarity_method cosine

# FB15k-237
python main.py --data ./data/FB15k-237/ --output_folder ./checkpoints/fb/out/  --similarity_method cosine 

```


