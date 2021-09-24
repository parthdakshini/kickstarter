# TO RUN THE APP:
#	* use command: "streamlit run streamlit_template.py"
import pandas as pd
import streamlit as st
import plotly.express as px


st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 90%;
        padding-top: 5rem;
        padding-right: 5rem;
        padding-left: 5rem;
        padding-bottom: 5rem;
    }}
    img{{
    	max-width:40%;
    	margin-bottom:40px;
    }}
</style>
""",
        unsafe_allow_html=True,
    )


# here is how to create containers
header_container = st.container()
stats_container = st.container()	

with header_container:
	# for example a logo or a image that looks like a website header
	#st.image('image.png')
	# Font size
	st.title("Kickstarter")
	st.header("Welcome!")
	st.subheader("It works")
	st.write("This is for font size")


# Another container
with stats_container:
	#You import datasets like you always do with pandas
	data = pd.read_csv('kickstarter_project\kickstarter_data_with_features.csv')
	
    #  You can work with data, change it and filter it
	ID = ['All'] + data['id'].unique().tolist()
	currency_trailing_code = ['All'] + data['currency_trailing_code'].unique().tolist()
	
    #collecting input from the user
	# collect input using free text
	# the input of the user will be saved to the variable called "text_input"
	text_input = st.text_input("You can collect free text input from the user", 'Something')

    # collect input using a list of options in a drop down format
	st.write('Or you can ask the user to select an option from the dropdown menu')
	s_station = st.selectbox('what column do you want to see', ID, key='ID')

	# display the collected input
	st.write('You selected the station: ' + str(s_station))

	# you can filter/alter the data based on user input and display the results in a plot
	st.write('And display things based on what the user has selected')
	if s_station != 'All':
		display_data = data[data['start station name'] == s_station]

	else:
		display_data = data.copy()

	# display the dataset in a table format
	st.write(display_data)


	# here is a different way of collecting data, namely multiple selection
	st.write('It is possible to have multiple selections too.')
	multi_select = st.multiselect('Which start station would you like to see?',ID, key='start_station', default=[])


	# creating columns inside a container
	bar_col, pie_col = st.columns(2)

	# in order to display things inside columns, replace the st. with the column name when creating components
	# preparing data to display on pie chart
	user_type = data['usertype'].value_counts().reset_index()
	user_type.columns = ['user type','count']






	# Creating plots and charts


	# preparing data to display in a bar chart
	start_location = data['start station name'].value_counts()

	# don't forget to add titles to your plots
	pie_col.subheader('no of id')

	# This is an example of a plotly pie chart
	fig = px.pie(user_type, values='count', names = 'user type', hover_name='user type')


	fig.update_layout(showlegend=False,
		width=400,
		height=400,
		margin=dict(l=1,r=1,b=1,t=1),
		font=dict(color='#383635', size=15))

	fig.update_traces(textposition='inside', textinfo='percent+label')

	# after creating the chart, we display it on the app's screen using this command
	pie_col.write(fig)