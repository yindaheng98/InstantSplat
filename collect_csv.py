import pandas as pd
import os

def collect_csv(folder_path, frame_indices):
    total_df = []

    # Step 1: Read all CSVs
    for frame_index in frame_indices:
        filename = os.path.join(folder_path, f"frame{frame_index}/ours_1000", "quality.csv")
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            total_df.append(df)
        else:
            print(f"Warning: File not found: {filename}")

    if not total_df:
        print("No valid CSV files found.")
        return None, None

    # Step 2: Concatenate all data
    total_df = pd.concat(total_df, ignore_index=True)

    # Step 3: Compute mean per name
    per_name_avg = total_df.groupby("name", as_index=False).mean(numeric_only=True)

    # Step 4: Weighted average (weighted by number of occurrences per name)
    counts = total_df["name"].value_counts().reindex(per_name_avg["name"]).values
    weighted_avg = (
        per_name_avg[["psnr", "ssim", "lpips"]] * counts[:, None]
    ).sum() / counts.sum()

    weighted_avg = weighted_avg.to_dict()

    return per_name_avg, weighted_avg

if __name__ == "__main__":
    results, sum = collect_csv("output/unaligned_frames3", range(0, 20))
    print(results)
    print("summary")
    print(sum)