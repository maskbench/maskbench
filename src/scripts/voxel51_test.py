import fiftyone as fo

# The directory containing the source images
data_path = "/datasets/ted-talks/0137_curiosity_10s"

# Import the dataset
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.VideoDirectory,
    dataset_dir=data_path,
    name="0137_curiosity_10s"
)

session = fo.launch_app(dataset, auto=False)
session.open()
session.wait()