# OPIM 5512 Craigslist Auto Listings Scraper and Price Prediction Pipeline
# * Mark Kulaga

This repo contains an automated pipeline that scrapes data from Craigslist car listings and makes price predictions.

I modified the baseline structure and files of the repo to expand the pipeline. Now, it scrapes data from Craigslist, extracts structured features with an LLM, trains a tuned GradientBoosting model, and then syncs predictions and interpretability artifacts back to the results folder in the repo. All this runs hourly on Google Cloud Platform.

## Detailed Additions
### Some of the specific additions I made to the repo for A08 are:
* Extending the LLM schema to capture transmission, drivetrain, fuel type, engine cylinders, condition, color, body typr, title status, and lcoation fileds such as City, State, and ZIP Code.
* Modifying train-dt to replace the existing model with a more complex one using GridSearchCV hyperparameter tuning
* Adding error metrics, permutation importance, and PDPs as synced model artifacts
* Adding my Model Analysis notebook to the repo root folder as a way of redundant access in case the links for it on colab do not work

## Important Changes to Repo Layout
* cloud_function/extractor-llm-poc/ — LLM extraction logic
* cloud_function/materialize-llm/ — JSONL to CSV aggregation
* cloud_function/train-dt/ — Updated model training and evaluation
* .github/workflows/ — deployment and sync-data.yml workflows
* results/ — accumulated model outputs from each hourly run