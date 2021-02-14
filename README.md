# Recommender Systems 2020 Challenge

This repo contains the code and the data used in the [Recommender Systems 2020 Challenge](https://www.kaggle.com/c/recommender-system-2020-challenge-polimi/leaderboard) @ Politecnico di Milano.<br> All the algorithms are in the [src](/src) folder and most of them are forked from the [course repository](https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi), which contains basic implementations of many recommenders and utility code.

I ended up with the following placement:

1. **Public leaderboard:** 5/66
2. **Private leaderboard:** 8/66

## Added functionalities

I added a couple of changes and useful extra code:

1) **GPU - Cython MF_IALS:** <br> The original [MF_IALS](/src/MatrixFactorization/IALSRecommender.py) algorithm was quite slow, so I implemented a faster version which can leverage on GPU using the [implicit library](https://github.com/benfred/implicit). I adapted the code provided in the implicit library to match the same interface of the course repository, thus I was able to use the already implemented evaluator, data strctures ecc... with little extra effort. The multithread CPU implementation allowed me to move from a 10 minutes per fit to about 30 seconds per fit, which is a huge improvement in performances.

2) **N score hybrid recommender**: <br> I extended the original [hybrid score recommender](src/KNN/ItemKNNScoresHybridRecommender.py) (which merges only two recommenders) to an arbitrary number of number of recommender. The code can be found in [GeneralizedMergedHybridRecommender.py](/src/Hybrid/GeneralizedMergedHybridRecommender.py), which merges the output scores from the recommenders, and in [GeneralizedSimilarityMergedHybridRecommender.py](/src/Hybrid/GeneralizedSimilarityMergedHybridRecommender.py), which merges the similarity matrices from item-based recommenders, applying a topK selection on the resulting matrix.

## Best model

My best model merges three different algoritms:
1) [MF_IALS](src/Implicit/FeatureCombinedImplicitALSRecommender.py) 
2) [RP3_Beta](src/GraphBased/RP3betaRecommender.py) 
3) [SLIM_ElasticNet](src/SLIM_ElasticNet/SLIMElasticNetRecommender.py) 

Each one of the above mentioned algorithms has been trained using the **Feature Combined** technique, which basically consists in merging the URM (user rating matrix) and the ICM (item content matrix) together and then training the model on this new matrix.

The resulting merged hybrid has been tuned both on the whole dataset and on the 25% users with the highest profile length, leveraging this specialization in a switching hybrid.

## My final presentation

[RecSys Presentation](https://github.com/Alexdruso/RecSysChallenge2020/raw/main/RecSys%20presentation.pdf) 

# FAQ
This section aims at helping future students with possible FAQ and problems we faced during the competition.

1. **I don't know where to start, how I do choose my recommender/model?** <br> Unfortunately there isn't any previous knowledge that tells you for sure which recommender will be the best, you have to try and see how they behave. Based on our experience, we suggest to first try the various recommenders and look for the best ones. In my case, I noticed from the very beginning that RP3_beta and graph based algorithms worked pretty well. Once you have found your "top tier" algorithms, you can start messing around with them. You can try to merge them using scores, similarity matrices, feature merging ecc... 

2. **Come on, is it really all trial and error?**: <br> More or less, yes, at least for the competition. Data is anonymized to avoid "cheating", so students can't just find an already existing solution and/or use extra data retrieved from the internet. However in general some "tricks" can be used. <br>In our challenge, we were given interactions with books and text tokens but, since the data are anonymized, it is impossible to perform text analysis, genre grouping, correlation between text and popularity ecc... However in real cases, these methods are commonly used. 

3. **Ok, so what did you do?** <br> As I already mentioned, at the very beginning I tested all the various algorithms we studied and I found out that graph based worked pretty well. So we decided to use this one as a "baseline" and we tried different combinations. 

4. **Some algorithms are almost impossible to fit, am I doing something wrong?** <br> Fortunately/unfortunately not, it is normal. Some algorithms (such as SLIM_ElasticNet, MF_IALS in the original implementation ecc...) take really A LOT of time and resources. SLIM_ElasticNet took 20 minutes per fit on a i7 3770k processor using the multithread implementation. Considering that these algorithms have to be tuned, doing just 50 trials took 50x20 minutes, which is quite a lot of time. I suggest exoloting some free AWS instances to tune the most challenging algorithms :)

5. **Well, I have just i3 my laptop, how am I supposed to join this challenge??** <br> You can use online services such as Kaggle, Google Colab or Google Cloud Platform. 

7. **Can I write you?** <br> Sure. You can open an issue on GitHub, write me on [LinkedIn](https://www.linkedin.com/in/alessandro-sanvito/) or ask on Telegram (@alexduso). Feel free to contact me, I don't bite :) 

Credits: thanks to Maurizio Ferrari Dacrema and prof. Paolo Cremonesi for the support in the challenge and thanks to Mattia Suricchio for the beautiful README template.
