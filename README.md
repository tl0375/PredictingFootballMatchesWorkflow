# PredictingFootballMatchesWorkflow
An extension of the previous predicting football matches project, achieved with a hybrid machine learning model to incorporate various forms of data. This version incorporates a website as a version of UI to view the predictions for upcoming matches, updating weekly with github actions.
##The Files
- Model.py - Python code showing the training (and limited testing for devlopment purposes) of the machine learning model used to predict football match outcomes using historical data. The model uses a hybrid approach concatenating the final predictions from a LSTM , MLP, and XGBoost algorithm to analyse patterns and generate win/draw/loss probabilities for upcoming matches. The LSTM portion of the model handles individual match data such as shots and goals, creating sequences over the past 10 games to analyse and draw predictions from. The MLP handles team rating statistics such as the previously mentioned ELO, and a binary flag that identifies promoted teams to aid in predictions. Finally, the XGBoost algorithm analyses a team's form over a number of timeframes to try and extract further meaningful insigths for overall prediction.
- Project Report.pdf - the accompanying report for the intial research and development of the above model as required as partof my Capstone Project at the University of York, graded at a First Class Honours level (76%).
- Update Dataset.py - scrapes https://www.football-data.co.uk/data.php and downloads the latest results .csv files and combines them with the existing dataset to add data form any matches that have been played since the last update.
- Fixture Scrape.py - Scrapes the upcoming fixtures with the use of the API from from https://www.football-data.org for each of the selected leagues (Premier League, La Liga, Serie A, Ligue 1 and Bundesliga) before storing in a .csv file.
- StandardiseFixtures.py - fixes some naming inconsistencies between these newly scraped fixtures and results data in not only column names but some of the data within the columns such as team names.
- Testing.py - Tests the upcoming fixtures from the newly created and cleaned fixtures dataset on model weights from the already trained model (saved in the testing folder), before creating a .csv file that contains the results of the predictions - percentage probabilities of each team winning a particular fixture. Also creates a record of the time of the last data update, saved in last_updated.txt within the website folder.
- football-predictor-ui - this folder provides all the files neccessary for the creation of the website to display the results, featuring the index.html file with corresponiding CSS and javascript files as well as the created reults.csv and a record of the last data update in last_updated.txt.

## The Workflow
- Update Dataset.py --> updated the dataset.csv file
- Fixture Scrape.py --> creates the fixtures.csv
- StandardiseFixtures.py --> take fixtures.csv and clean, outputting fixtures_clean.csv
- Testing --> take fixtures_clean.csv and outputs results.csv stored in the website (football-predictor-ui) folder.
- Display website
