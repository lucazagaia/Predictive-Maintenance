import pandas as pd

def load_label_dict(label_csv_path):
    """
    Load stream_labels.csv and convert it to a lookup dictionary.
    Returns:
        A dictionary with keys (device_id, timestamp_begin, timestamp_end) → failure label (0 or 1)
    """
    df = pd.read_csv(label_csv_path)
    
    # Convert string failure labels to numeric (0 = Working, 1 = Failure)
    def map_failure_label(failure_str):
        if isinstance(failure_str, str):
            if failure_str.lower() == 'working':
                return 0
            else:
                return 1  # Any non-working state is considered a failure
        else:
            # If it's already numeric, try to convert it
            return int(failure_str)
    
    label_dict = {
        (str(row['device_id']), int(row['timestamp_begin']), int(row['timestamp_end'])): map_failure_label(row['failure'])
        for _, row in df.iterrows()
    }

    return label_dict
