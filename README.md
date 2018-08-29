# Fraud Detection 

## Movitation & Background

Events website is a convenient platform for people to post information about their events. However, there are also increasing fraud activities on the event websites.

An event website is curious to know how can we use Machine Learning to predict an event posted live is a fraud or not.

Here is a visualization of the fraud activities happening in North America on this website.

![geo](https://github.com/liyouzhang/Fraud_Detection/blob/dev/pics/geo.jpg)

## How to use the fraud detection product
![frontend_dashborad](https://github.com/liyouzhang/Fraud_Detection/blob/dev/pics/frontend_dashboard.jpg)

- For Investigators:
    - an interactive web page to flag potential fraud


- For Product Managers / Business Stakeholders:
    - Monitor trends of fraud on event and user level:
        - events that were published on user creation date
        - events missing description
        - users' age, type, email domain etc
    - Potential bans at the event posting stage

## Architecture

![architecture](https://github.com/liyouzhang/Fraud_Detection/blob/dev/pics/architecture.jpg)

## Model & Methodology

### Key challenges:
- imbalanced data
- big data

### Model comparison

![models](https://github.com/liyouzhang/Fraud_Detection/blob/dev/pics/model_selection.jpg)

### Cost Benefit Analysis

![cost](https://github.com/liyouzhang/Fraud_Detection/blob/dev/pics/Cost_benefit.jpg)

## Tech Stack
- AWS
- Flask
- Python
- Sklearn
