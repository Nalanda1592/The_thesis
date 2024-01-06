For thesis code:-
If someone wants to run this app in their local, following steps need to be followed:-
1. Add parent_dir variable already instantiated, in front of relative path of all csv files.
3. Use terminal command "streamlit run visualization.py" to run the application locally.

The_thesis Streamlit app:-

Steps for whole dataset:-

1. Click on the dataset you want to use first.

2. Click on the button refering to the topic you want to explore, and start chatting about that topic to the chatbot.

3. Give feedback in the form of thumbs up/down  on each answer and please give detailed feedback in the textbox, and then submit. The Thumbs are located at the bottom of the query textbox in mild grey outline color. Please locate it carefully.

Steps for single result:-

1. Click on the dataset you want to use first.

2. Ask in the chatbox "How to predict salary"

3. Copy the question template from the answer, modify using your own values and send the query

4. After you get the results, you can ask questions about this prediction's SHAP explanations

5. Use this question or something similar to get the SHAP explanation : "what are the most important features for a single prediction according to shap values?" . Use keywords "single prediction","shap"

Example dataset questions:-
1. How many columns are there in the dataset?
2. What is the max value of numeric columns in the dataset?
3. What is the average actual salary in the dataset?
4. What is the average actual salary based on gender in the dataset?
5. How many females are there in the dataset?
6. What is the average actual salary for 30 year olds in the dataset?
7. What is the average predicted salary for 32 year old phd holders in the dataset? 

Example SHAP Explanation questions:-
1. What are the most important features of the dataset?
2. What are the most important features of the 10th observation of the dataset? Use whole sentence?
3. What is shap?

Example Fairlearn fairness questions:-
1. What can be interpreted from the fairlearn results?
2. What metrices are used to calculate the results and what do they mean?
3. How can the results be improved?

Example Unfairness Mitigation questions:-
1. What can be inferred from the unfairness mitigation results?
2. What are the metrices and technique used to do mitigation?
3. What do they mean in context of Fairlearn fairness library?

!!!!!WARNINGS!!!!!

1. Delete the text in the query textbox before you switch to another operation like changing the dataset or choosing another topic through button.

2. The LLM gives irrelevant results to appropriate questions sometimes, depending on the refined query. Please rephrase your query with keyworks like "use tool", "use whole sentence as input" etc.

3. Different tools have been used to answer different queries. Please note the tool names. If answers are irrelevant, rephrase the query with the tool name:-
	i)		PredictedDatasetTool() - answers query about whole dataset and its predictions on test data

	ii)		ShapExplainPredictedResultTool() - answers query about Shap explanations of the    above predictions

	iii)	FairlearnFairnessTool() - Measures fairness of the model

	iv)		UnfairnessMitigationTool() - Mitigates unfairness of the model

	v)		SinglePredictionTool() - 

	vi)		SingleExplanationTool() 




