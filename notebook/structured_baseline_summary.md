# Structured Baseline Summary

## Completed Modeling Steps

Steps 1 through 4 are complete for the first modeling pass.

1. Created a `price_tier` target with three classes: low, medium, and high.
2. Built a clean structured modeling dataframe using vehicle listing fields.
3. Created one stratified train/test split.
4. Trained and evaluated a dense neural network structured baseline model.

## Data Source

The preferred source file is `structured/datasets/listings_llm.csv`, but that file is currently stored in Mark's private Google Cloud Storage bucket. Since Andrew's account does not have read access, this first pass uses the latest GitHub `preds.csv` artifact from the `results` folder.

This should be replaced with the full `listings_llm.csv` file before final submission if Mark can provide access or export the CSV.

## Structured Baseline Results

| Model | Input Type | Accuracy | Weighted F1 | Main Takeaway |
|---|---|---:|---:|---|
| Structured Dense Baseline | Structured listing fields | 0.7059 | 0.6909 | Strong on low and high tiers, weaker on medium tier |

## Confusion Matrix Takeaway

The structured model correctly classified 14 of 17 low-price listings and 15 of 17 high-price listings. It struggled most with the medium tier, correctly classifying 7 of 17 medium-price listings.

The model did not make the most severe pricing mistake: it did not classify any low-price cars as high, or any high-price cars as low.

## Next Steps

1. Replace the fallback `preds.csv` file with the full `listings_llm.csv` dataset.
2. Start the text-only LSTM model using title and description.
3. Confirm whether the dataset contains image URLs or image paths.
4. If image data is not available, extend the scraper pipeline to collect image URLs or download the first image per listing.
