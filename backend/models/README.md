# 模型缓存目录

`*.pkl` 文件属于本地训练缓存，不提交到 Git。模型缺失时，预测接口会从
`backend/data/air_quality_cities.csv` 自动训练并生成对应缓存。