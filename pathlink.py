from typing import DefaultDict
from numpy.core.fromnumeric import size
import streamlit as st
import pandas as pd
import numpy as np
from k_means_constrained import KMeansConstrained
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from toolz.itertoolz import groupby

#Suppress warning about streamlit.file_uploader
st.set_option('deprecation.showfileUploaderEncoding', False)


#################################### APP CONTENT STARTS ####################################

'''
# PathLink Dashboard

Upload your form data here and select the number of students you want per group! We'll take care of the rest, so that you don't have to worry about matching students :)
'''


def load_data(filepath, header, first_row_is_header=False):
    df = pd.read_csv(filepath, header=None)
    df.columns = header
    df.drop(['Timestamp'], axis=1, inplace=True)
    if first_row_is_header:
        df.drop(0, axis=0, inplace=True)
    df.set_index('ID', inplace=True)
    return df


@st.cache
def make_groups(df, total_students, students_per_group):
    df.drop(["Name", "Email"], axis=1, inplace=True)
    df = pd.get_dummies(df, columns=['Year', 'Interests'], drop_first=False)

    def encode(df):

        def skill_encoder(df):
            for i in range(len(df.iloc[:, 2])):
                if df.iloc[i, 2] == 4:
                    df.iloc[i, 2] = 2
                elif df.iloc[i, 2] == 5:
                    df.iloc[i, 2] = 1
                    
        def availability_encoder(df):
            for i in range(len(df.iloc[:, 1])):
                if df.iloc[i, 1] == "00:00 - 6:00":
                    df.iloc[i, 1] = 0
                elif df.iloc[i, 1] == "6:00 - 12:00":
                    df.iloc[i, 1] = 1
                elif df.iloc[i, 1] == "12:00 - 18:00":
                    df.iloc[i, 1] = 2
                elif df.iloc[i, 1] == "18:00 - 24:00":
                    df.iloc[i, 1] = 3
            
        def timezone_encoder(df):
            for i in range(len(df.iloc[:, 0])):
                if df.iloc[i, 0] == "GMT–8 (Pacific Time)":
                    df.iloc[i, 0] = 0
                elif df.iloc[i, 0] == "GMT–6 (CST)":
                    df.iloc[i, 0] = 1
                elif df.iloc[i, 0] == "GMT–5 (EST)":
                    df.iloc[i, 0] = 2
                elif df.iloc[i, 0] == "GMT–3 (South America)":
                    df.iloc[i, 0] = 3
                elif df.iloc[i, 0] == "GMT+0 (GMT)":
                    df.iloc[i, 0] = 4
                elif df.iloc[i, 0] == "GMT+1 (CET)":
                    df.iloc[i, 0] = 5
                elif df.iloc[i, 0] == "GMT+3 (Eastern Europe/Middle East)":
                    df.iloc[i, 0] = 6
                elif df.iloc[i, 0] == "GMT+5 (South Asia)":
                    df.iloc[i, 0] = 7
                elif df.iloc[i, 0] == "GMT+8 (East Asia)":
                    df.iloc[i, 0] = 8
                elif df.iloc[i, 0] == "GMT+10 (Australia)":
                    df.iloc[i, 0] = 9
                elif df.iloc[i, 0] == "GMT+12 (New Zealand)":
                    df.iloc[i, 0] = 10
            df["Timezone"] = df["Timezone"].astype(int)

        df["Timezone"] = df["Timezone"].astype(str)
        timezone_encoder(df)
        
        df["Availability"] = df["Availability"].astype(str)
        availability_encoder(df)
        
        df["Skill"] = df["Skill"].astype(int)
        skill_encoder(df)
        
        return df

    df = encode(df)

    n_groups = total_students//students_per_group
    min_students=0
    max_students=0

    if total_students%students_per_group == 0:
        min_students = students_per_group
        max_students = students_per_group
    else:
        n_groups += 1
        min_students = total_students - (students_per_group*(total_students//students_per_group))
        max_students = students_per_group


    groups = KMeansConstrained(
        n_clusters=n_groups, 
        size_min=min_students, 
        size_max=max_students
    )
    groups.fit_predict(df)
    
    return groups.labels_.astype(int).tolist()



def df_to_group_dict(df):
    '''
    Takes in a dataframe with a ["Groups"] column and returns a dictionary having
    keys as the group number and values as the group dataframe.
    '''
    group_set = set([i for i in df["Groups"]])

    group_df_dict = DefaultDict(pd.DataFrame)

    for i in range(len(group_set)):
        temp_df = df[df["Groups"] == i]

        group_df_dict[i] = temp_df
        
        print(temp_df)

    return group_df_dict



st.sidebar.title("PathLink ©")

st.sidebar.markdown("### Upload your data here:")
uploaded_file =  st.sidebar.file_uploader(label="Supported formats: .csv", type=["csv"])


if uploaded_file is not None:
    '''
    ## Here's your uploaded data.
    '''

    header = ["Timestamp", "ID","Name","Email","Year","Timezone","Availability","Skill","Interests"]
    df = load_data(uploaded_file, header, first_row_is_header=True)

    st.sidebar.markdown("### Enter the number of students you want to have per group.")
    students_per_group = st.sidebar.number_input("", min_value=1, max_value=len(df))
    dr_dataset = st.sidebar.checkbox(label="Show data representation for the whole dataset.")
    dr_groups = st.sidebar.checkbox(label="Show data representation for separate groups", value=True)

    st.write(df)
    
    total_students = len(df)

    st.markdown(f"Total number of students = {total_students}")

    with st.spinner("Making groups..."):
        df["Groups"] = make_groups(df.copy(), total_students, students_per_group)
        groups = df_to_group_dict(df)
    
    if dr_dataset:
        with st.spinner('Drawing visualizations...'):
    
            fig1 = px.scatter_3d(df, x="Availability", y="Timezone", z="Year",
            color="Groups")
            fig1.update_layout(
                font_size=10, 
                title={
                    'text': "Overall Class Distribution",
                    'y':0.9,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'}
            )

            st.write(fig1)

            fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Skill vs Year Distribution", "Timezone Distribution"), specs=[[{"type": "xy"}, {"type": "domain"}]])
            

            combos = list(zip(df.Year, df.Skill))
            weight_counter = Counter(combos)

            weights = [3*weight_counter[(df.Year[i], df.Skill[i])] for i, _ in enumerate(df.Year)]

            fig2.add_trace(
                go.Scatter(x=df.Year, y=df.Skill, mode="markers", marker_size=weights),
                row=1, col=1
            )


            labels = []
            sizes = []
            timezone_dict = defaultdict(int)
            for i in df.Timezone:
                timezone_dict[i] += 1
            for i, j in timezone_dict.items():
                labels.append(i)
                sizes.append(j)

            fig2.add_trace(
                go.Pie(values=sizes, labels=labels), 
                row=1, col=2
            )

            fig2.update_layout(showlegend=False)
            fig2.update_xaxes(showgrid=False, zeroline=False, title_text="Year of Study", row=1, col=1)
            fig2.update_yaxes(showgrid=False, zeroline=False, title_text="Skill Level", row=1, col=1)
            st.write(fig2)

    '''
    ## Groups have been formed! 

    You can select from the slider to see a particular group along with some statistics about it.
    '''

    n_groups = total_students//students_per_group
    if total_students%students_per_group != 0:
        n_groups += 1
    group_number = st.slider(
        label= "", 
        min_value=1, 
        max_value=n_groups
    )

    st.write(f"Group {group_number} ({len(groups[group_number - 1])} students)")
    st.write(groups[group_number - 1])
    

    if dr_groups:
        with st.spinner('Drawing visualizations...'):
            fig3 = make_subplots(rows=2, cols=2, 
                subplot_titles=("Skill vs Year Distribution", "Availability Distribution", "Timezone Distribution", "Interests Distribution"), 
                specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "domain"}, {"type": "domain"}]]
            )
            
            #Plot Skill vs Year distribution
            combos = list(zip(groups[group_number - 1].Year, groups[group_number - 1].Skill))
            weight_counter = Counter(combos)
            weights = [10*weight_counter[(groups[group_number - 1].Year[i], groups[group_number - 1].Skill[i])] for i, _ in enumerate(groups[group_number - 1].Year)]

            fig3.add_trace(
                go.Scatter(x=groups[group_number - 1].Year, y=groups[group_number - 1].Skill, mode="markers", marker_size=weights),
                row=1, col=1
            )

            # Plot Availability Distribution
            labels = []
            sizes = []
            availability_dict = defaultdict(int)
            for i in groups[group_number - 1].Availability:
                availability_dict[i] += 1
            for i, j in availability_dict.items():
                labels.append(i)
                sizes.append(j)

            fig3.add_trace(
                go.Bar(x=labels, y=sizes),
                row=1, col=2
            )

            #Plot Timezone Distribution
            labels = []
            sizes = []
            timezone_dict = defaultdict(int)
            for i in groups[group_number - 1].Timezone:
                timezone_dict[i] += 1
            for i, j in timezone_dict.items():
                labels.append(i)
                sizes.append(j)

            fig3.add_trace(
                go.Pie(values=sizes, labels=labels), 
                row=2, col=1
            )


            #Plot Interests Distribution
            labels = []
            sizes = []
            timezone_dict = defaultdict(int)
            for i in groups[group_number - 1].Interests:
                timezone_dict[i] += 1
            for i, j in timezone_dict.items():
                labels.append(i)
                sizes.append(j)
            fig3.add_trace(
                go.Pie(values=sizes, labels=labels), 
                row=2, col=2
            )


            fig3.update_layout(showlegend=False, height=800)
            fig3.update_xaxes(showgrid=False, zeroline=False, title_text="Year of Study", row=1, col=1)
            fig3.update_yaxes(showgrid=False, zeroline=False, title_text="Skill Level", row=1, col=1)
            fig3.update_xaxes(showgrid=False, zeroline=False, title_text="Time Interval (in EST)", row=1, col=2)
            fig3.update_yaxes(showgrid=False, zeroline=False, title_text="Number of Students", row=1, col=2)
            st.write(fig3)
    




            # fig = plt.figure(figsize=(20,20))
            # gs = GridSpec(nrows=2, ncols=2)
            # fig.tight_layout(pad=5.0)
            # plt.rcParams.update({'font.size': 20})



            # #Plot availability bar chart
            # labels = []
            # sizes = []
            # availability_dict = defaultdict(int)
            # for i in groups[group_number - 1].Availability:
            #     availability_dict[i] += 1
            # for i, j in availability_dict.items():
            #     labels.append(i)
            #     sizes.append(j)
            # ax0 = fig.add_subplot(gs[0, 0])
            # ax0.set_title("Availability distribution (Times in EDT)")
            # ax0.bar(labels, sizes)
            # ax0.set_xlabel("Time intervals in EDT")
            # ax0.set_ylabel("Number of Students")


            # #Plot year of study vs skill level
            # combos = list(zip(groups[group_number - 1].Year, groups[group_number - 1].Skill))
            # weight_counter = Counter(combos)

            # weights = [20*20**weight_counter[(groups[group_number - 1].Year[i], groups[group_number - 1].Skill[i])] for i, _ in enumerate(groups[group_number - 1].Year)]

            # ax1 = fig.add_subplot(gs[0, 1])
            # ax1.set_title("Year of study vs skill level distribution")
            # ax1.scatter(groups[group_number - 1].Year, groups[group_number - 1].Skill, s=weights)
            # ax1.set_xlabel("Year of study")
            # ax1.set_ylabel("Skill")


            # #Plot timezone pie chart
            # labels = []
            # sizes = []
            # timezone_dict = defaultdict(int)
            # for i in groups[group_number - 1].Timezone:
            #     timezone_dict[i] += 1
            # for i, j in timezone_dict.items():
            #     labels.append(i)
            #     sizes.append(j)
            # ax2 = fig.add_subplot(gs[1, :])
            # ax2.set_title("Timezone distribution")
            # ax2.pie(sizes, labels=labels, autopct='%1.1f%%',
            #         shadow=True, startangle=90)
            # ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.


            # st.pyplot(fig)