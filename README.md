
<div align = "center" style = "font-family: Bangla Sangam MN;">
<font size= 22px><b> Word2Vec Movie Sentiment Classification  <br>--  with Skipgram and CBOW  Implementation
 <font></b>

<hr/>
</div>
<p style = "margin-left :4ch; margin-right:5ch; font-family: Bangla Sangam MN; color: #000; font-size: 18px">
<b>Movie Review Sentiment Classification</b>
</p>

<blockquote><p align = "center"><i> <font size = 17px face= "Impact" color = 'gray'>" </font>the fly-on-the-wall method used to document rural french school life is a refreshing departure from the now more prevalent technique of the docu-makers being a visible part of their work . B+</i></p>
  <footer align = "center"  >— <font size = 4px>review for <a href="https://www.imdb.com/title/tt0318202/?ref_=fn_al_tt_1">Être et avoir</a></font></footer></blockquote>

<p style = "margin-left :4ch; margin-right:5ch; font-family: Helvetica Neue; color: #909090; font-size: 16px; line-height:1.4">

Reading the  review above for the French movie Être et avoir (2002), a documentary directed by Nicolas Philibert and winner of the 2002 European Film Award in the European Documentary category, do you think the author's attitude to this film is positive or negative? It's hard to judge a people's opinion simply based on one sentence but NLP is able to achieve this goal. 
<br><br>
This project explores the power of word-vectors to do multi-classification with simply a multi-logistic regression model. The reviews have 5 sentiment levels: --, -, neutral, +, ++ 
<font color = "MediumVioletRed"> (see chart 1) </font>. I first implemented Skipgram and CBOW (Continuous Bag of Words) with numpy to train word-vectors on my training corpus and then test the trained word-vectors on the movie sentiment classification task with sklearn logistic regression. We can try fancier models but the goal here is to test how well we train the word-vectors and compare it against <a href="https://nlp.stanford.edu/projects/glove/">GloVe (Global Vectors) </a> -- word vectors trained  on aggregated global word-word co-occurrence statistics . I compared the performance of my self-trained word vectors with GloVe that was trained with a larger corpus. <br><br>
</p>
<div align = "center">
<img src = "https://img1.wsimg.com/isteam/ip/2d33dabb-b536-484f-ac3e-25fe02574c07/183fd173-fba1-467a-b907-cf9113e2a83c.png/:/cr=t:0%25,l:0%25,w:100%25,h:100%25/rs=w:622,h:311,cg:true">
</div>
<div align = "center">
<img src = "https://img1.wsimg.com/isteam/ip/2d33dabb-b536-484f-ac3e-25fe02574c07/420ccc51-0634-4622-9263-03c5e94b2a4f.png/:/cr=t:0%25,l:0%25,w:100%25,h:100%25/rs=h:650,cg:true"; width = 412px>
</div>

<div align = "center">
<img src = 
"https://img1.wsimg.com/isteam/ip/2d33dabb-b536-484f-ac3e-25fe02574c07/ace2f00c-0845-4abd-a1b1-d94d095be5ec.png/:/cr=t:0%25,l:0%25,w:100%25,h:100%25/rs=h:650,cg:true"; width = 600px>
 
 </div>
<br><br>
<font color = "MediumVioletRed"> Chart 3</font> visualises the word vectors after  de-meaning and then projects the vectors on 2 dimensions after PCA dimension reduction. For certain words, similar words are closer to each other in terms to cosine similarity (their angles with respect to the (0, 0) origin in the graph. 
</p>
