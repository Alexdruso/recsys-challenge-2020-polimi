# Recommender Systems 2020 Challenge

This repo contains the code and the data used in the [Recommender Systems 2020 Challenge](https://www.kaggle.com/c/recommender-system-2020-challenge-polimi/leaderboard) @ Politecnico di Milano.<br> All the algorithms are in the [src](/src) folder and most of them are forked from the [course repository](https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi), which contains basic implementations of many recommenders and utility code.

I ended up with the following placement:

1. **Public leaderboard:** 5/66
2. **Private leaderboard:** 8/66

## Added functionalities

I added a couple of changes and useful extra code:

1) **GPU - Cython MF_IALS:** <br> The original [MF_IALS](/MatrixFactorization/algorithm/IALSRecommender.py) algorithm was quite slow, so we implemented a faster version which leaverages on GPU using the [implicit library](https://github.com/benfred/implicit). We adapted the code provided in the implicit library to match the same interface of our course repository, thus we were able to use the already implemented evaluator, data strctures ecc... with little extra effort. The GPU implementation allowed us to move from a 10 minutes per fit to about 30 seconds per fit, which is a huge improvement in performances.

2) **N score hybrid recommender**: <br> We extended the original [hybrid score recommender](/KNN/ItemKNNScoresHybridRecommender.py) (which merges only two recommenders) to an arbitrary number of number of recommender. The code can be found in [ItemKNNScoresHybridNRecommender.py](/KNN/ItemKNNScoresHybridNRecommender.py). <br> **NB:** there are other classes like [ItemKNNScoresHybrid5Recommender.py](/KNN/ItemKNNScoresHybrid5Recommender.py) which are "noise" from our various experiments. As the name suggest, this hybrid merges 5 recommenders. We higly suggest to use the generalized version [ItemKNNScoresHybridNRecommender.py](/KNN/ItemKNNScoresHybridNRecommender.py) which is simpler, cleaner and more flexible.

## Best model

Our best model merges three different algoritms:
1) [MF_IALS](/MatrixFactorization/algorithm/IALSRecommender.py) 
2) [RP3_Beta](/GraphBased/RP3betaRecommender.py) 
3) [SLIM_ElasticNet](/SLIM_ElasticNet/SLIMElasticNetRecommender.py) 

Each one of the above mentioned algorithms has been trained using the **Feature Merging** technique, which basically consists in merging the URM (user rating matrix) and the ICM (item content matrix) together and then training the model on this new matrix.

The best models can be found in [Best Models](/Challenge_2020/Best_models), in particular the code of the above mentioned model is [MF_IALS+rp3+Slim_elasticNet_featuremerge_0.09917](/Challenge_2020/Best_models/MF_IALS+rp3+Slim_elasticNet_featuremerge_0.09917_test.ipynb). The other models in this folder, even if they have a higher test score, were achieving worse performances on the private leaderboard. 

## Our final presentation

[RecSys Presentation](https://github.com/mattiasu96/Recommender-Systems-Challenge/blob/main/RecSys%20Presentation.pdf) 

# FAQ
This section aims at helping future students with possible FAQ and problems we faced during the competition.

1. **I don't know where to start, how I do choose my recommender/model?** <br> Unfortunately there isn't any previous knowledge that tells you for sure which recommender will be the best, you have to try and see how they behave. Based on our experience, we suggest to first try the various recommenders and look for the best ones. In our case, we noticed from the very beginning that RP3_beta and graph based algorithms worked pretty well. Once you have found your "top tier" algorithms, you can start messing around with them. You can try to merge them using scores, similarity matrices, feature merging ecc... 

2. **Come on, is it really all trial and error?**: <br> More or less, yes, at least for the competition. Data is anonymized to avoid "cheating", so students can't just find an already existing solution and/or use extra data retrieved from the internet. However in general some "tricks" can be used. <br>In our challenge, we were given interactions with books and text tokens but, since the data are anonymized, it is impossible to perform text analysis, genre grouping, correlation between text and popularity ecc... However in real cases, these methods are commonly used. 

3. **Ok, so what did you do?** <br> As I already mentioned, at the very beginning we tested all the various algorithms we studied and we found out that graph based worked pretty well. So we decided to use this one as a "baseline" and we tried different combinations. At a certain point we added a PureSVD recommender to our already existing graph based model, and we saw an incredible spike in performances. We ended up having a merge of p3alpha, rp3beta, pureSVD and userKNN for the first deadline, which was good, but not enough, it kind of overfitted. <br> During the course we were introduced to the **feature merging** technique. We tried a lot of methods to perform feature selection and analysis, without any success (and a lot of blue screens, check the performance question), however feature merging allowed us to exploit the available features in our training process. We noticed that with this technique, we greatly improved the performances of our "standalone" Rp3Beta, so we decided to apply this method also to other recommenders. We found out that MF_IALS performed really well with feature merging, and we had already noticed that graph based + matrix factorization was a pretty good combo, thus we decided to put MF_IALS together with the Rp3Beta and check if our reasoning was correct. Indeed it was! 

4. **Some algorithms are almost impossible to fit, am I doing something wrong?** <br> Fortunately/unfortunately not, it is normal. Some algorithms (such as SLIM_ElasticNet, MF_IALS in the original implementation ecc...) take really A LOT of time and resources. SLIM_ElasticNet took 20 minutes per fit on a i7 3770k processor using the multithread implementation. Considering that these algorithms have to be tuned, doing just 50 trials took 50x20 minutes, which is quite a lot of time. 

5. **Well, I have just i3 my laptop, how am I supposed to join this challenge??** <br> You can use online services such as Kaggle, Google Colab or Google Cloud Platform. We ended up using Google Colab scripts with an autoclicker on our mobile phone to run the code without getting disconnected for inactivity during night. Not the most beautiful solution, but you have to work with what you have ;) 

6. **Do you have any examples of how to set-up those platforms?** <br> Yep. Here you can find an example on Kaggle https://www.kaggle.com/mattiasurricchio/tuning-userknn-feature-merge where we tuned a UserKNN. https://www.kaggle.com/mattiasurricchio/baserecommenders this is the dataset needed to use the course code (obviously it will not be updated for future versions of the course). The main idea is to download the course repo from GitHub, zip it and then upload it as dataset. Then you will be able to use it on Kaggle. <br> The same holds for Google Colab, you can find an example here: https://colab.research.google.com/drive/1PwMkHZpCpAbCzPPgihOjdD3T1fOMHnpO?usp=sharing

7. **Can I write you?** <br> Sure. You can open an issue on GitHub, write us on LinkedIn (https://www.linkedin.com/in/mattiasurricchio/ https://www.linkedin.com/in/arcangelo-pisa-8166a366/) or ask on the Telegram group of the course (we will probably be there for a while). Feel free to contact us, we don't bite :) 
