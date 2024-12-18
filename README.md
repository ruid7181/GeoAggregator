# Implementation of the paper 'GeoAggregator: An Efficient Transformer Model for Geo-spatial Tabular Data'

## Introduction
Modeling geospatial tabular data with deep learning has become a promising alternative to traditional statistical and machine learning approaches. 
However, existing deep learning models often face challenges related to scalability and flexibility as datasets grow. 
To this end, this paper introduces GeoAggregator, an efficient and lightweight algorithm based on transformer architecture designed specifically for geospatial tabular data modeling. 
GeoAggregators explicitly account for spatial autocorrelation and spatial heterogeneity through Gaussian-biased local attention and global positional awareness. 
Additionally, we introduce a new attention mechanism that uses the Cartesian product to manage the size of the model while maintaining strong expressive power. 
We benchmark GeoAggregator against spatial statistical models, XGBoost, and several state-of-the-art geospatial deep learning methods using both synthetic and empirical geospatial datasets. 
The results demonstrate that GeoAggregators achieve the best or second-best performance compared to their competitors on nearly all datasets. 
GeoAggregator’s efficiency is underscored by its reduced model size, making it both scalable and lightweight. 
Moreover, ablation experiments offer insights into the effectiveness of the Gaussian bias and Cartesian attention mechanism, providing recommendations for further optimizing the GeoAggregator’s performance.

