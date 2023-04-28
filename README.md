# User Review Summarization in Russian

This project researches the use of currently available architectures for the Russian language opinion summarization.

For every method in its folder you can see 'train.sh' file that reproduces training from scratch and instructions from the authors.

## Data 
The data was parsed from Tripadvisor, Lamoda and Rendez-Vous online services and can be downloaded [here](https://drive.google.com/drive/folders/1sT22ORCqC_JCdMpIlqA1SmIZHDoYRKP1?usp=sharing). For reproduction the data should be extracted in the root directory.
The golden set was marked up manually and split into dev and test splits. The data for the markup was taken from the parsed reviews and stratified by rating.
