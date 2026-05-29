import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns



st.set_page_config(page_title="Intelligent Book Recommendations", layout="wide")
st.markdown("""
<style>
h1 { font-weight: 800; color: #111; }
h2 { font-weight: 700; }
p  { font-size: 18px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("📚 Audible Insights: Intelligent Book Recommendations")
menu = st.sidebar.radio(
    "Go to",
    [
        "📘Introduction",
        "📊EDA & Insights",
        "📖Book Recommendation"
    ]
)
st.title("📚 Audible Insights: Intelligent Book Recommendations")
if menu == "📘Introduction":
    st.header("Introduction")
    st.write("With the large number of books available today, finding the right book can be difficult for readers. This project develops a book recommendation system that suggests similar books based on genres, descriptions, and authors. The system uses machine learning and NLP techniques and is deployed using Streamlit for an interactive user experience..")
    st.header("Objectives:")
    st.write("🟪 Clean and prepare the book datasets.")

    st.write("🟪 Analyze book data using EDA to understand trends.")

    st.write("🟪 Use NLP techniques to find similarities between books.")

    st.write("🟪 Build a Streamlit app to recommend books.")

    with st.expander("🎯View Project Architecture"):

        st.markdown("""
        **Project Workflow**

        Dataset  
        ↓  
        Data Cleaning  
        ↓  
        Exploratory Data Analysis (EDA)  
        ↓  
        Feature Engineering  
        ↓  
        NLP (TF-IDF)  
        ↓  
        Cosine Similarity  
        ↓  
        Recommendation System  
        ↓  
        Clustering  
        ↓  
        Model Evaluation  
        ↓  
        Streamlit Application
        """)


if menu == "📊EDA & Insights":
    st.header("Exploratory Data Analysis")

    df = pickle.load(open(r"C:\Users\ARUL RAJ HARESH\OneDrive\Desktop\yashma\books.pkl","rb"))
    similarity = pickle.load(open(r"C:\Users\ARUL RAJ HARESH\OneDrive\Desktop\yashma\similarity.pkl","rb"))


    question = st.selectbox(
        "Select EDA question",
        [
           "Q1 Most popular genres in the dataset",
           "Q2 Authors have the highest-rated books",
           "Q3 Average rating distribution across books",
           "Q4 Price distribution of books",
           "Q5 How do ratings vary between books with different review counts",
           "Q6 Which books are frequently clustered together based on descriptions",
           "Q7 How does genre similarity affect book recommendations",
           "Q8 Top Authors by Average Ratings",
           "Q9 Which combination of features provides the most accurate recommendations",
           "Q10 A new user likes science fiction books",
           "Q11 Recommend thriller books",
           "Q12 Hidden gems books"
           
        ]  
    ) 
    if question =="Q1 Most popular genres in the dataset": 
        
        genres = df['Genres_text'].str.split()

        genres = genres.explode()

        top_genres = genres.value_counts().head(10)

        fig, ax = plt.subplots()

        top_genres.plot(kind='barh', ax=ax)

        ax.set_title("Top Genres")
        ax.set_xlabel("Number of Books")

        st.pyplot(fig)

    if question =="Q2 Authors have the highest-rated books": 
        

        author_rating = df.groupby('Author')['Rating'].mean().sort_values(ascending=False)

        fig, ax = plt.subplots()

        author_rating.head(10).plot(kind='bar', ax=ax)

        ax.set_title("Top Authors by Average Rating")
        ax.set_ylabel("Average Rating")

        st.pyplot(fig)


    if question =="Q3 Average rating distribution across books":
        fig, ax = plt.subplots()
        sns.histplot(df['Rating'], bins=20, ax=ax)
        st.pyplot(fig)

    

    if question =="Q4 Price distribution of books":

        fig, ax = plt.subplots()

        sns.histplot(df['Price'], bins=20, ax=ax)

        ax.set_title("Price Distribution of Books")
        ax.set_xlabel("Price")
        ax.set_ylabel("Number of Books")

        st.pyplot(fig)

    if question =="Q5 How do ratings vary between books with different review counts":

        fig, ax = plt.subplots()

        sns.scatterplot(x='Rating', y='Number of Reviews', data=df, ax=ax)

        ax.set_title("Ratings vs Number of Reviews")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Number of Reviews")

        st.pyplot(fig)

    if question =="Q6 Which books are frequently clustered together based on descriptions":

        cluster_books = df.groupby("cluster")["Book Name"].apply(list)

        for c, books in cluster_books.items():
            st.subheader(f"Cluster {c}")
            for b in books[:5]:
                st.write(b)


    if question =="Q7 How does genre similarity affect book recommendations":
        genres = df['Genres_text'].str.split()

        genres = genres.explode()

        top_genres = genres.value_counts().head(10)

        fig, ax = plt.subplots()

        top_genres.plot(kind='barh', ax=ax)

        ax.set_title("Top Genres")
        ax.set_xlabel("Number of Books")

        st.pyplot(fig)  


    if question =="Q8 Top Authors by Average Ratings":

        author_rating = df.groupby("Author")["Rating"].mean().sort_values(ascending=False)

        fig, ax = plt.subplots()

        author_rating.head(10).plot(kind="bar", ax=ax)

        ax.set_title("Top Authors by Average Rating")

        st.pyplot(fig)

    if question =="Q9 Which combination of features provides the most accurate recommendations":

    

        st.write("""
        The recommendation system performs best when combining multiple textual features.
        
        The most effective combination includes:
        
        - Book Description
        - Genres
        - Author
        
        These features were merged into a single column called **combined_features**.
        TF-IDF vectorization was applied to convert the text into numerical vectors, and cosine similarity was used to find similar books.
        """)

    if question =="Q10 A new user likes science fiction books":
        sci_fi_books = df[df['Genres_text'].str.contains("science fiction", case=False, na=False)]

        top_sci_fi = sci_fi_books.sort_values(by="Rating", ascending=False).head(5)

        top_sci_fi[["Book Name","Author","Rating"]]


    if question =="Q11 Recommend thriller books":

        thriller_books = df[df['Genres_text'].str.contains("thriller", case=False, na=False)]

        top_thrillers = thriller_books.sort_values(by="Rating", ascending=False).head(5)

        st.dataframe(top_thrillers[["Book Name","Author","Rating"]])

    if question =="Q12 Hidden gems books":

        hidden_gems = df[(df["Rating"] > 4.5) & (df["popularity_score"] < df["popularity_score"].median())]

        hidden_gems = hidden_gems.sort_values(by="Rating", ascending=False).head(5)

        st.dataframe(hidden_gems[["Book Name","Author","Rating","popularity_score"]])
            
if menu == "📖Book Recommendation":
    
    

    df = pickle.load(open(r"C:\Users\ARUL RAJ HARESH\OneDrive\Desktop\yashma\books.pkl","rb"))
    similarity = pickle.load(open(r"C:\Users\ARUL RAJ HARESH\OneDrive\Desktop\yashma\similarity.pkl","rb"))

    def recommend(book):

        index = df[df["Book Name"] == book].index[0]

        scores = list(enumerate(similarity[index]))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        books = []

        for i in scores[1:6]:
            books.append(df.iloc[i[0]]["Book Name"])

        return books
    st.header("📚 Book Recommendation")

    selected_book = st.selectbox(
        "Select a Book",
        df["Book Name"].values
    )

    if st.button("Recommend"):

        recommendations = recommend(selected_book)

        for book in recommendations:
            st.write(book)
    def evaluate_precision(book):

        recommended_books = recommend(book)

        input_genre = df[df["Book Name"] == book]["Genres_text"].values[0]

        if input_genre == "unknown" or input_genre == "":
            return "Genre not available for evaluation"

        recommended_genres = df[df["Book Name"].isin(recommended_books)]["Genres_text"]

        relevant = 0

        for g in recommended_genres:
            if any(word in str(g) for word in input_genre.split()):
                relevant += 1

        precision = relevant / len(recommended_books)

        return precision

    precision = evaluate_precision(selected_book)
    st.write("Precision:", precision)
