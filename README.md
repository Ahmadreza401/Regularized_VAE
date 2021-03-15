# Regularized_VAE
Regularized Variational Auto-encoder (VAE) for Constrained Reconstruction of Time-series

Constrained reconstruction is mainly involved withidentifying  and  controlling  the  underlying  factorsof  generation  that  contribute  in  the  production  ofsamples  of  a  dataset.   Although  many  researcheshave  studied  this  problem  on images, only few have focused on time-series data. For  time-series  data,   a  model  capable  of  constrained  reconstruction  should  be  able  to  identify factors like amplitude, seasonality, frequency, trendand so on as the basic elements that make differ-ent  time-series.    The  model  should  also  be  ableto  control  this  factors  when  generating  new  samples.   Such  model  enables  unsupervised  filtering of  dataset  samples  and  crafting  similar,  manipulated datasets where the samples have the desiredattributes. It can have many applications in datasetsaugmentation and targeted data generation.  In this project  we  propose  a  new  VAE-based  model  that uses disentanglement learning for capturing the underlying factors of generation for time-series data.

Author: Ahmad Asgharian Rezaei
All rights reserved.
