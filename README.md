# OPIM 5512/5509 Craigslist Auto Listings Scraper and Price Tier Classification
# * Mark Kulaga, Andrew Ghali, Joseph Berkowitz

This repo contains an automated pipeline that scrapes Craigslist car listings, extracts structured listing data, and supports a final deep learning project for classifying vehicles into low, medium, and high price tiers.

We modified the original scraper and prediction pipeline to support a multi-modal modeling workflow. The updated pipeline now collects structured vehicle fields, listing text, and image URLs, then materializes a final modeling dataset used for structured, text, image, and combined deep learning models.

## Detailed Additions
### Some of the specific additions made to the repo include:
* Extending the LLM extraction schema to capture vehicle fields such as transmission, drivetrain, fuel type, engine cylinders, condition, color, body type, title status, and location fields.
* Adding `combined_text`, `combined_text_len`, and `has_combined_text` so listing text can be used in LSTM-based models.
* Preserving `image_url` values so listing photos can be used in image-based models.
* Creating a final modeling dataset at `data/final_modeling_listings.csv`.
* Building and comparing structured baseline, text-only LSTM, image-only ConvNet, structured + text, and structured + text + image models.
* Adding model evaluation outputs including accuracy, weighted F1, classification reports, confusion matrices, and qualitative error analysis.

## Final Modeling Result
The best-performing model was the **Structured + Text** model, which combined structured vehicle fields with listing text.

* Accuracy: 0.7634
* Weighted F1: 0.7559

The main takeaway is that structured vehicle fields carried most of the predictive signal, while listing text added a modest improvement. Image data contained some signal on its own, but adding images to the full fused model did not improve performance on this small dataset.

## Important Changes to Repo Layout
* `cloud_function/extractor-llm-poc/` — LLM extraction logic and combined text creation
* `cloud_function/materialize-llm/` — JSONL to CSV aggregation for the final modeling dataset
* `cloud_function/scraper_cars/` — Craigslist scraping logic
* `.github/workflows/` — deployment and dataset sync workflows
* `data/` — final modeling dataset
* `notebook/` — modeling notebooks and summaries
* `results/` — synced outputs and modeling artifacts
