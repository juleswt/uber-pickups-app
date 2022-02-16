import streamlit as st
import pandas as pd
import plotly.express as px

from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# session_state definition
def goto_home_page(value):
    st.session_state.home_page = value
    goto_cluster_selection(0)
    goto_definition_model(0)

def goto_definition_model(value):
    st.session_state.definition_model = value
    goto_cluster_selection(0)

def goto_cluster_selection(value):
    st.session_state.cluster_selection = value

def choose_cluster_number():
    if st.session_state.cluster_user:
        st.session_state.cluster_number = st.session_state.cluster_user

# session_state initialisation
if "home_page" not in st.session_state:
    st.session_state["home_page"] = 0

if "definition_model" not in st.session_state:
    st.session_state["definition_model"] = 0

if "cluster_selection" not in st.session_state:
    st.session_state["cluster_selection"] = 0

if "cluster_number" not in st.session_state:
    st.session_state["cluster_number"] = 1

# just for debugging
# st.write(st.session_state["cluster_selection"])
# st.write(st.session_state["cluster_number"])

# dataset & model defintion
def build_df(file, year, month, hour):
    # github subfolder path: ("bloc-3/uber-pickups/src/" + file)
    
    df = pd.read_csv("src/" + file)
    df["Date/Time"] = pd.to_datetime(df["Date/Time"], format="%m/%d/%Y %H:%M:%S")
    df = df.loc[df["Date/Time"].dt.year == year, :]
    df = df.loc[df["Date/Time"].dt.month == month, :]
    df = df.loc[df["Date/Time"].dt.hour == hour, :]
    df["minute"] = df["Date/Time"].dt.minute
    return df

def auto_kmeans(days, df):
    model = []
    for count, elt in enumerate(days):
        name = list(elt.keys())[0]
        day = list(elt.values())[0]
        
        df_day = df.loc[df["Date/Time"].dt.dayofweek == day, :]
        df_minute = df_day.loc[:, "minute"]
        df_day = df_day.loc[:, ["Lat", "Lon"]]
        
        sc = StandardScaler()
        X = sc.fit_transform(df_day)
        
        sse = {}
        for i in range (1, 11):
            kmeans = KMeans(n_clusters=i, random_state=0)
            kmeans.fit(X)
            sse[i] = kmeans.inertia_
        
        kn = KneeLocator(x=list(sse.keys()), y=list(sse.values()), curve="convex", direction="decreasing").knee
        kmeans = KMeans(n_clusters=kn+1, random_state=0).fit(X)
        df_day["cluster"] = kmeans.labels_
        
        df_day = df_day[df_day.cluster != -1]
        df_day = pd.concat([df_day, df_minute], axis=1)
        df_day = df_day.assign(day=name)
        df_day["minute"] = df_day["minute"].astype(int)
        model.append(df_day)
    
    df1 = model[0]
    for i in range(len(model)-1):
        df1 = pd.concat([df1, model[i+1]], axis=0)
    
    df1.sort_values(by=["minute", "day"], ascending=True, inplace=True)
    df1.reset_index(drop=True, inplace=True)
    return df1

def dbscan(days, df, eps, min_samples, distance):
    model = []
    for count, elt in enumerate(days):
        name = list(elt.keys())[0]
        day = list(elt.values())[0]
        
        df_day = df.loc[df["Date/Time"].dt.dayofweek == day, :]
        df_minute = df_day.loc[:, "minute"]
        df_day = df_day.loc[:, ["Lat", "Lon"]]
        
        sc = StandardScaler()
        X = sc.fit_transform(df_day)
        
        db = DBSCAN(eps=eps, min_samples=min_samples, metric=distance).fit(X)
        df_day["cluster"] = db.labels_
        
        df_day = pd.concat([df_day, df_minute], axis=1)
        df_day = df_day.assign(day=name)
        df_day["minute"] = df_day["minute"].astype(int)
        model.append(df_day)
    
    df1 = model[0]
    for i in range(len(model)-1):
        df1 = pd.concat([df1, model[i+1]], axis=0)
    
    df1.sort_values(by=["minute", "day"], ascending=True, inplace=True)
    df1.reset_index(drop=True, inplace=True)
    return df1

def elbow_silhouette(day, df):
    df_day = df.loc[df["Date/Time"].dt.dayofweek == day, :]
    df_day = df_day.loc[:, ["Lat", "Lon"]]
    
    sc = StandardScaler()
    X = sc.fit_transform(df_day)
    
    # Within Cluster Sum of Square
    wcss = []
    k = []
    for i in range (1, 11):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        k.append(i)
    
    wcss_frame = pd.DataFrame(wcss)
    k_frame_elbow = pd.Series(k)
    
    sil = []
    k = []
    for i in range (2, 11):
        kmeans = KMeans(n_clusters= i, random_state = 0)
        kmeans.fit(X)
        sil.append(silhouette_score(X, kmeans.predict(X)))
        k.append(i)
    
    cluster_scores = pd.DataFrame(sil)
    return df_day, X, wcss_frame, k_frame_elbow, k, cluster_scores

def manu_kmeans(cluster, df, X):
    kmeans = KMeans(n_clusters=cluster, random_state=0).fit(X)
    df["cluster"] = kmeans.labels_
    return df

# APPLICATION START HERE...
st.set_page_config(page_title="Uber Pickups", layout="wide")

# mapping variables
file_list = ["uber-raw-data-apr-14.csv", "uber-raw-data-may-14.csv", "uber-raw-data-jun-14.csv", "uber-raw-data-jul-14.csv", "uber-raw-data-aug-14.csv", "uber-raw-data-sep-14.csv"]
hour_list = ["00h00", "01h00", "02h00", "03h00", "04h00", "05h00", "06h00", "07h00", "08h00", "09h00", "10h00", "11h00", "12h00", "13h00", "14h00", "15h00", "16h00", "17h00", "18h00", "19h00", "20h00", "21h00", "22h00", "23h00"]
mapping = {"apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "14.csv": 2014, "00h00": 0, "01h00": 1, "02h00": 2, "03h00": 3, "04h00": 4, "05h00": 5, "06h00": 6, "07h00": 7, "08h00": 8, "09h00": 9, "10h00": 10, "11h00": 11, "12h00": 12, "13h00": 13, "14h00": 14, "15h00": 15, "16h00": 16, "17h00": 17, "18h00": 18, "19h00": 19, "20h00": 20, "21h00": 21, "22h00": 22, "23h00": 23}
days = [{"day1": 0}, {"day2": 1}, {"day3": 2}, {"day4": 3}, {"day5": 4}, {"day6": 5}, {"day7": 6}]
day_list = ["Monday (day1)", "Tuesday (day2)", "Wednesday (day3)", "Thursday (day4)", "Friday (day5)", "Saturday (day6)", "Sunday (day7)"]

if st.session_state.home_page == 0:
    
    st.markdown("""
    ![](https://upload.wikimedia.org/wikipedia/commons/thumb/5/58/Uber_logo_2018.svg/360px-Uber_logo_2018.svg.png "UBER LOGO")
    
    ## Company's Description 📇
    
    > Uber is one of the most famous startup in the world. It started as a ride-sharing application for people who couldn't afford a taxi. Now, Uber expanded its activities to Food Delivery with Uber Eats package delivery, freight transportation and even urban transportation Jump Bike and Lime that the company funded.
    
    ## Project 🚧
    
    > One of the main pain point that Uber's team found is that sometimes drivers are not around when users need them. For example, a user might be in San Francisco's Financial District whereas Uber drivers are looking for customers in Castro. Eventhough both neighborhood are not that far away, users would still have to wait 10 to 15 minutes before being picked-up, which is too long. Uber's research shows that users accept to wait 5-7 minutes, otherwise they would cancel their ride. Therefore, Uber's data team would like to work on a project where their app would recommend hot-zones in major cities to be in at any given time of day.
    
    ## Goals 🎯
    
    > Uber already has data about pickups in major cities. Your objective is to create algorithms that will determine where are the hot-zones that drivers should be in. Therefore you will:
    
    - Create an algorithm to find hot zones
    - Visualize results on a nice dashboard
    
    ## Deliverable 📬
    
    To complete this project, your team should:
    
    - Have a map with hot-zones using any python library
    - You should at least describe hot-zones per day of week
    - Compare results with at least two unsupervised algorithms like KMeans and DBScan
    """)
    
    home_page_button = st.sidebar.button("APP", on_click=goto_home_page, args=[1])

if st.session_state.home_page == 1:
    
    expander_bar = st.expander("About")
    expander_bar.markdown("""
    * **Python libraries**: streamlit, pandas, plotly, kneed, sklearn
    * **Data Source**: [Uber Trip Data](https://full-stack-bigdata-datasets.s3.eu-west-3.amazonaws.com/Machine+Learning+non+Supervis%C3%A9/Projects/uber-trip-data.zip)
    * **Authors**: Jules Walbert
    """)
    
    home_page_button = st.sidebar.button("Home Page", on_click=goto_home_page, args=[0])
    
    if st.session_state.definition_model == 0:
        
        st.sidebar.write("---")
        definition_model_button = st.sidebar.button("KMeans / DBScan", on_click=goto_definition_model, args=[1])
        
        st.sidebar.write("---")
        st.sidebar.header("Select your configuration")
        
        file = st.sidebar.selectbox("File", file_list)
        hour = st.sidebar.selectbox("Hour", hour_list)
        method = st.sidebar.radio("Method", ("KMeans (Auto)", "DBScan", "KMeans (Manual)"), on_change=goto_cluster_selection, args=[0])
        
        if method == "DBScan":
            
            st.sidebar.write("---")
            st.sidebar.header("Set specific parameters")
            eps = st.sidebar.slider("Epsilon Distance", 0.0, 1.0, 0.2, 0.001)
            min_samples = st.sidebar.slider("Minimum Neighbors", 1, 100, 30, 1)
            distance = st.sidebar.select_slider("Distance calculation", options=["euclidean", "manhattan"])
        
        if method == "KMeans (Manual)":
            
            st.sidebar.write("---")
            st.sidebar.header("Set specific parameters")
            day = st.sidebar.selectbox("Day Of Week", day_list)
            day = day.split("(")[1].split(")")[0]
            for elt in days:
                try:
                    day = elt[day]
                except:
                    pass
        
        if st.session_state.cluster_selection == 0:
            
            st.write("Click to prepare the dataset and run the calculation")
            if st.button("Run calculaton"):
                
                month = mapping[file.split("-")[3]]
                year = mapping[file.split("-")[4]]
                hour = mapping[hour]
                
                if file and hour and method:
                    
                    df = build_df(file, year, month, hour)
                    st.write(f"The dataset contains {df.shape[0]} raws!")
                    
                    if method == "KMeans (Auto)":
                        
                        df1 = auto_kmeans(days, df)
                        fig = px.scatter_mapbox(df1, lat="Lat", lon="Lon", animation_group="minute", animation_frame="day", color="cluster", mapbox_style="carto-positron")
                        st.plotly_chart(fig)
                    
                    elif method == "DBScan":
                        
                        df2 = dbscan(days, df, eps, min_samples, distance)
                        fig = px.scatter_mapbox(df2, lat="Lat", lon="Lon", animation_group="minute", animation_frame="day", color="cluster", mapbox_style="carto-positron")
                        st.plotly_chart(fig)
                    
                    elif method == "KMeans (Manual)":
                        
                        df_day, X, wcss_frame, k_frame_elbow, k, cluster_scores = elbow_silhouette(day, df)
                        
                        elbow = px.line(wcss_frame, x=k_frame_elbow, y=wcss_frame.iloc[:,-1])
                        elbow.update_layout(yaxis_title="Inertia", xaxis_title="# Clusters", title="Inertia per cluster")
                        st.plotly_chart(elbow)
                        
                        sil = px.bar(data_frame=cluster_scores, x=k, y=cluster_scores.iloc[:, -1])
                        sil.update_layout(yaxis_title="Silhouette Score", xaxis_title="# Clusters", title="Silhouette Score per cluster")
                        st.plotly_chart(sil)
                        
                        st.write("---")
                        change = st.button("Choose Cluster Number", on_click=goto_cluster_selection, args=[1])
                
                else:
                    st.success("Please select a file, an hour and a method")
        
        if st.session_state.cluster_selection == 1:
            
            st.write("---")
            cluster = st.number_input("Cluster Number", 1, 10, 3, 1, on_change=choose_cluster_number, key="cluster_user")
            
            month = mapping[file.split("-")[3]]
            year = mapping[file.split("-")[4]]
            hour = mapping[hour]
            
            df = build_df(file, year, month, hour)
            df_day, X, wcss_frame, k_frame_elbow, k, cluster_scores = elbow_silhouette(day, df)
            
            df3 = manu_kmeans(st.session_state["cluster_number"], df_day, X)
            clusters = px.scatter_mapbox(df3[df3.cluster != -1], lat="Lat", lon="Lon", color="cluster", mapbox_style="carto-positron")
            st.plotly_chart(clusters)
            
            st.write("---")
            change = st.button("Back to method selection", on_click=goto_cluster_selection, args=[0])
    
    if st.session_state.definition_model == 1:
        
        st.sidebar.write("---")
        definition_model_button = st.sidebar.button("Configuration Selection", on_click=goto_definition_model, args=[0])
        
        col1, col2 = st.columns((1, 1))
        col1.markdown("""
        #### **KMEANS**

        *Definition:*
        
        > - Separate sample in n groups of equal variance
        > - Distance between data points and center
        > - top-down algorithm, starts from sets ends to observations
        
        *Steps:*

        1. Initialize K clusters
        2. Calculate centroïds
        3. Assign data points to the closest centroïd (distance calculation)
        4. Calculate centroïds again and then assign data points again
        5. Clusters are stable when no observation is moving from a cluster to another
        
        *Elbow:*

        > - Check if data points within a cluster are close from their centroid
        > - The more clusters we get the less different the inertia (wcss) is going to be
        > - Tells us how homogeneous the values are within a cluster
        
        *Silhouette:*

        > - Check if clusters are far from each other
        > - Silhouette = 1 = clusters far from each other
        > - Tells us how distinct our clusters are from each other
        """)
        
        col2.markdown("""
        #### **DBSCAN**

        *Definition:*
        
        > - Create cluster based on how close each sample are from each other
        > - Density of data points in space
        > - Bottom-up algorithm, starts from observations ends to sets

        *Steps:*

        1. Define min_samples and epsilon
        2. Define if core_sample or not
        3. Redo step 2
        4. Get clusters

        *Parameters:*

        > - min _sample: how many observations to  create a core sample
        > - epsilon: maximum distance to define an observation as part of a sample
        """)