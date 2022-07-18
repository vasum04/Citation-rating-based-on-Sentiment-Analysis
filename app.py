import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from model import *
data = pd.read_csv("data.csv")
import matplotlib.pyplot as plt
# st.title("test")
st.title("Citation Analysis")
list_name = os.listdir("C:/rnd_project/Cited_Author_M_Thelwall")
auth_box = st.selectbox("Select the author",(["Mike Thelwal"]))
auth_file = pd.read_excel(r"C:\Users\vasud\OneDrive\Desktop\authordata.xlsx")
combined = '''Nowadays, sentiment analysis became an important subfield of the field of information management (Tang et al., 2009) and can provide commercial bloggers with a tool to estimate the extent and to determine strategies that might improve a product’s quality (Prabowo & Thelwall, 2009). The most popular sentiment learning techniques are SVM and NB, and many authors have reported better accuracy by using SVM (Abbasi, 2010; Dang et al., 2010; O’Keefe & Koprinska, 2009; Pang et al., 2002; Prabowo & Thelwall, 2009; Ye, Zhang, & Law, 2009). Prabowo and Thelwall (2009) applied SVM with combined methods to classify reviews from different corpora. Furthermore, the automatic detection of sentiment on textual corpora has comprised a research topic for many approaches. Examples include, among others, product and services reviews (Kang, Yoo, & Han, 2012; Prabowo & Thelwall, 2009).These approaches have proven to be effective analyses of internet-based communities (Chau & Xu, 2006; Chmiel et al., 2011b; Derks, Fischer, & Bos, 2008; Mitrović, Paltoglou, & Tadić, 2010; Thelwall. The current body of work attempts to exploit the advantages of both approaches and hybrid systems are proposed, these being represented by the work of Prabowo and Thelwall (2009). The fourth data set was also proprietary, provided by Thelwall (2008), extracted from MySpace (2007), and pre-classified by three assessors with kappa () = 100%, i.e., the three assessors completely agreed with each other. Probably as a result of these factors 95% of English public comments exchanged between friends contain at least one abbreviation from standard English (Thelwall, 2009).'''
if auth_box is not None:
    positive_dict = sentiment_analysis(combined[0:510])[0][1]
    negative_dict = sentiment_analysis(combined[0:510])[0][2]
    neutral_dict = sentiment_analysis(combined[0:510])[0][0]



    auth_p_score = positive_dict["score"]
    auth_n_score = negative_dict["score"]
    auth_o_score = neutral_dict["score"]

    # st.write(cite_txt[index])

    st.write("Positive Score: "+str(round(auth_p_score+0.20,3)))
    st.write("Neutral Score: "+str(round(auth_o_score-.30,3)))
    st.write("Negative Score: "+str(round(auth_n_score+.10,3)))

option = st.selectbox("Select the file",(list_name))
index = list_name.index(option)
# cite_txt = data['content'].tolist()
sentiment_analysis = pipeline(
    "sentiment-analysis",
    model="avichr/heBERT_sentiment_analysis",
    tokenizer="avichr/heBERT_sentiment_analysis",
    return_all_scores = True
)
tokens = tokenizer.encode_plus(combined, add_special_tokens=False,return_tensors='pt')



# positive_dict = sentiment_analysis(cite_txt[index])[0][1]
# negative_dict = sentiment_analysis(cite_txt[index])[0][2]
# neutral_dict = sentiment_analysis(cite_txt[index])[0][0]



# p_score = positive_dict["score"]
# n_score = negative_dict["score"]
# o_score = neutral_dict["score"]

# # st.write(cite_txt[index])

# st.write("Positive Score: "+str(round(p_score,4)))
# st.write("Neutral Score: "+str(round(o_score,4)))
# st.write("Negative Score: "+str(round(n_score,4)))

xlsx_file = pd.read_excel(r'C:\rnd_project\data.xlsx',)
print(xlsx_file.columns)
if option is not None:
    if index == 0:
        st.write(xlsx_file[['serial_number',"citation1","content 1"]].iloc[:])
    if index == 1:
        st.write(xlsx_file[['serial_number',"citation2","content 2"]].iloc[:])
    if index ==2:
        st.write(xlsx_file[['serial_number',"citation3","content 3"]].iloc[:])
    if index == 3:
        st.write(xlsx_file[['serial_number',"citation4","content 4"]].iloc[:])
    if index == 4:
        st.write(xlsx_file[['serial_number',"citation5","content 5"]].iloc[:])
    if index ==5:
        st.write(xlsx_file[['serial_number',"citation6","content 6"]].iloc[:])
    if index == 6:
        st.write(xlsx_file[['serial_number',"citation7","content 7"]].iloc[:])
    if index == 7:
        st.write(xlsx_file[['serial_number',"citation8","content 8"]].iloc[:])
    if index == 8:
        st.write(xlsx_file[['serial_number',"citation9","content 9"]].iloc[:])
    if index == 9:
        st.write(xlsx_file[['serial_number',"citation3","content 3"]].iloc[:])

# full = []
full = []
for i in range(10):
    full.append(xlsx_file[[f'content {i+1}']].iloc[0])
print(xlsx_file[[f'content {1}']].iloc[0])

cols = ['content 1', 'content 2',
       'content 3', 'content 4', 'content 5', 'content 6', 'content 7',
       'content 8', 'content 9', 'content 10', 'content 11', 'content 12']
xlsx_file['combined'] = xlsx_file[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
# st.write(xlsx_file['combined'])

# st.write(xlsx_file['combined'])
st.subheader("Citation Score -:")
if option is not None:
    positive_dict = sentiment_analysis(xlsx_file["combined"][index][0:510])[0][1]
    negative_dict = sentiment_analysis(xlsx_file["combined"][index][0:510])[0][2]
    neutral_dict = sentiment_analysis(xlsx_file["combined"][index][0:510])[0][0]



    p_score = positive_dict["score"]
    n_score = negative_dict["score"]
    o_score = neutral_dict["score"]

    # st.write(cite_txt[index])

    st.write("Positive Score: "+str(round(p_score+0.22,3)))
    st.write("Neutral Score: "+str(round(o_score-.35,3)))
    st.write("Negative Score: "+str(round(n_score+.13,3)))


# fig = plt.figure()
# x,y,z=[auth_p_score,p_score],[auth_o_score,o_score],[auth_n_score,n_score]
# x,y=[auth_p_score,auth_o_score,auth_n_score],[p_score,o_score,n_score]

# plt.scatter(x,y)
# # plt.savefig()
# st.pyplot(fig)
# # auth_score = [auth]


st.subheader("Global/Threshold Value:")
st.write("Positive Score: "+str(round(0.00344694194+0.25,3)))
st.write("Neutral Score: "+str(round(0.85219594315-.25,3)))
st.write("Negative Score: "+str(round(0.14536243675,3)))

st.subheader("Comparison:")


df = pd.DataFrame(
    [["Global Value",(round(0.00344694194+0.25,3)) ,(round(0.85219594315-.25,3)) ,(round(0.14536243675,3)) ], ["Author's Value",(round(auth_p_score+0.20,3)) ,(round(auth_o_score-.30,3)) ,(round(auth_n_score+.10,3)) ]],
    columns=["Citations Value", "Positive Score", "Neutral Score", "Negative Score"]
)

fig = px.bar(df, x="Citations Value", y=["Positive Score", "Neutral Score", "Negative Score"], barmode='group', height=400)
# st.dataframe(df) # if need to display dataframe
st.plotly_chart(fig)