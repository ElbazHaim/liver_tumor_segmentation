import argparse
from liver_tk.processing.process_image import process_and_save_files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process and save 3D medical images and segmentation masks."
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to the CSV file containing volume and segment paths.",
    )
    parser.add_argument(
        "output_image_dir", type=str, help="Directory to save processed volume files."
    )
    parser.add_argument(
        "output_segment_dir",
        type=str,
        help="Directory to save processed segment files.",
    )
    parser.add_argument(
        "--target_depth", type=int, required=True, help="Target depth for the images."
    )
    parser.add_argument(
        "--window_level",
        type=float,
        required=True,
        help="Window level for image windowing.",
    )
    parser.add_argument(
        "--window_width",
        type=float,
        required=True,
        help="Window width for image windowing.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    process_and_save_files(
        args.csv_file,
        args.output_image_dir,
        args.output_segment_dir,
        args.target_depth,
        args.window_level,
        args.window_width,
    )


if __name__ == "__main__":
    main()
