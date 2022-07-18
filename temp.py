# import streamlit as st
# import pandas as pd
# import numpy as np
# from model import *
# data = pd.read_csv("data.csv")
# st.title("test")
# user_input = st.text_input("label goes here", "")
# # st.selectbox("select the paper :",["1","2"])
# import os
# list_name = os.listdir("C:/rnd_project/Cited_Author_M_Thelwall")
# option = st.selectbox("Select the file",(list_name))
# cite_txt = data['content'].tolist()
# # print(cite_txt)
# index = list_name.index(option)
# sentiment_analysis = pipeline(
#     "sentiment-analysis",
#     model="avichr/heBERT_sentiment_analysis",
#     tokenizer="avichr/heBERT_sentiment_analysis",
#     return_all_scores = True
# )

# positive_dict = sentiment_analysis(cite_txt[index])[0][1]
# negative_dict = sentiment_analysis(cite_txt[index])[0][2]
# neutral_dict = sentiment_analysis(cite_txt[index])[0][0]



# p_score = positive_dict["score"]
# n_score = negative_dict["score"]
# o_score = neutral_dict["score"]












'''Nowadays, sentiment analysis became an important subfield of the field of information management (Tang et al., 2009) and can provide commercial bloggers with a tool to estimate the extent and to determine strategies that might improve a product’s quality (Prabowo & Thelwall, 2009). The most popular sentiment learning techniques are SVM and NB, and many authors have reported better accuracy by using SVM (Abbasi, 2010; Dang et al., 2010; O’Keefe & Koprinska, 2009; Pang et al., 2002; Prabowo & Thelwall, 2009; Ye, Zhang, & Law, 2009). Prabowo and Thelwall (2009) applied SVM with combined methods to classify reviews from different corpora. Furthermore, the automatic detection of sentiment on textual corpora has comprised a research topic for many approaches. Examples include, among others, product and services reviews (Kang, Yoo, & Han, 2012; Prabowo & Thelwall, 2009).These approaches have proven to be effective analyses of internet-based communities (Chau & Xu, 2006; Chmiel et al., 2011b; Derks, Fischer, & Bos, 2008; Mitrović, Paltoglou, & Tadić, 2010; Thelwall. The current body of work attempts to exploit the advantages of both approaches and hybrid systems are proposed, these being represented by the work of Prabowo and Thelwall (2009). The fourth data set was also proprietary, provided by Thelwall (2008), extracted from MySpace (2007), and pre-classified by three assessors with kappa () = 100%, i.e., the three assessors completely agreed with each other. Probably as a result of these factors 95% of English public comments exchanged between friends contain at least one abbreviation from standard English (Thelwall, 2009).'''
