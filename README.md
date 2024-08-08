# User Review Summarization in Russian

Prisiazhniuk, A., Malykh, V. (2024). User Review Summarization in Russian. In: Ignatov, D.I., et al. Recent Trends in Analysis of Images, Social Networks and Texts. AIST 2023. Communications in Computer and Information Science, vol 1905. Springer, Cham. https://doi.org/10.1007/978-3-031-67008-4_7

This project researches the use of currently available architectures for the Russian language opinion summarization. We explored unsupervised and weakly-supervised settings, the usage of PlanSum, AceSum, QT and SemAE models.

For every method in its folder you can see 'train.sh' file that reproduces training from scratch and instructions from the authors. The models were trained as in the corresponding papers with small deviations for the adaptation to the Russian language data.

## Data 
The data was parsed from Tripadvisor
online service and can be downloaded [here](https://drive.google.com/drive/folders/1sT22ORCqC_JCdMpIlqA1SmIZHDoYRKP1?usp=sharing) along with model checkpoints. For reproduction the data should be extracted in the root directory.
The golden set was marked up manually and split into dev and test splits. The data for the markup was taken from the parsed reviews and stratified by rating.
