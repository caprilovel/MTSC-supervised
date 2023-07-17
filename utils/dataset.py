import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from einops import rearrange, reduce, repeat
from utils.utils import string2boolean
from sktime.datasets._data_io import load_from_tsfile
from sklearn.preprocessing import LabelEncoder
import time 
import re


def series_to_nd(m):
    series_len = len(m[0][0])
    train_nd = np.empty((len(m), len(m[0]), series_len))
    for i in range(len(m)):
        for j in range(len(m[0])):
            train_nd[i][j] = np.array(m[i][j])
    return train_nd

def load_config_from_tsfile(
    full_file_path_and_name,
):
    """Load config from a .ts file into a dict.

    Parameters
    ----------
    full_file_path_and_name: str
        The full pathname of the .ts file to read.
    
    Returns
    -------
    Dict
        Contains a dictionary of the data format, but not the data content.
    """
    data_config = {}
    data_started = False
    # Parse the file
    with open(full_file_path_and_name, "r", encoding="utf-8") as file:
        for line in file:
            # Strip white space from start/end of line and change to
            # lowercase for use below
            line = line.strip().lower()
            
            # Empty lines are valid at any point in a file
            if line:
                # Check if this line contains metadata
                # Please note that even though metadata is stored in this
                # function it is not currently published externally
                if line.startswith("@"):
                    if data_started:
                        raise IOError("metadata must come before data")
                    
                    if not line.startswith("@data"):
                        datastrip = line.split(" ")
                        if datastrip[0] == '@classlabel' and datastrip[1] =='true':
                            data_config['classlabel'] = datastrip[2:]
                        else:
                            config_name = datastrip[0].lstrip("@")    
                            data_config[config_name] = string2boolean(datastrip[1])
                    
                            
    return data_config


def load_from_tsfile2dataframe(
    full_file_path_and_name,
    return_separate_X_and_y=True,
    replace_missing_vals_with="NaN",
):
    """Load data from a .ts file into a Pandas DataFrame.

    Parameters
    ----------
    full_file_path_and_name: str
        The full pathname of the .ts file to read.
    return_separate_X_and_y: bool
        true if X and Y values should be returned as separate Data Frames (
        X) and a numpy array (y), false otherwise.
        This is only relevant for data that
    replace_missing_vals_with: str
       The value that missing values in the text file should be replaced
       with prior to parsing.

    Returns
    -------
    DataFrame (default) or ndarray (i
        If return_separate_X_and_y then a tuple containing a DataFrame and a
        numpy array containing the relevant time-series and corresponding
        class values.
    DataFrame
        If not return_separate_X_and_y then a single DataFrame containing
        all time-series and (if relevant) a column "class_vals" the
        associated class values.
    """
    # Initialize flags and variables used when parsing the file
    metadata_started = False
    data_started = False

    has_problem_name_tag = False
    has_timestamps_tag = False
    has_univariate_tag = False
    has_class_labels_tag = False
    has_data_tag = False

    previous_timestamp_was_int = None
    prev_timestamp_was_timestamp = None
    num_dimensions = None
    is_first_case = True
    instance_list = []
    class_val_list = []
    line_num = 0
    # Parse the file
    with open(full_file_path_and_name, "r", encoding="utf-8") as file:
        for line in file:
            # Strip white space from start/end of line and change to
            # lowercase for use below
            line = line.strip().lower()
            # Empty lines are valid at any point in a file
            if line:
                # Check if this line contains metadata
                # Please note that even though metadata is stored in this
                # function it is not currently published externally
                if line.startswith("@problemname"):
                    # Check that the data has not started
                    if data_started:
                        raise IOError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len == 1:
                        raise IOError("problemname tag requires an associated value")
                    # problem_name = line[len("@problemname") + 1:]
                    has_problem_name_tag = True
                    metadata_started = True
                elif line.startswith("@timestamps"):
                    # Check that the data has not started
                    if data_started:
                        raise IOError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len != 2:
                        raise IOError(
                            "timestamps tag requires an associated Boolean " "value"
                        )
                    elif tokens[1] == "true":
                        timestamps = True
                    elif tokens[1] == "false":
                        timestamps = False
                    else:
                        raise IOError("invalid timestamps value")
                    has_timestamps_tag = True
                    metadata_started = True
                elif line.startswith("@univariate"):
                    # Check that the data has not started
                    if data_started:
                        raise IOError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len != 2:
                        raise IOError(
                            "univariate tag requires an associated Boolean  " "value"
                        )
                    elif tokens[1] == "true":
                        # univariate = True
                        pass
                    elif tokens[1] == "false":
                        # univariate = False
                        pass
                    else:
                        raise IOError("invalid univariate value")
                    has_univariate_tag = True
                    metadata_started = True
                elif line.startswith("@classlabel"):
                    # Check that the data has not started
                    if data_started:
                        raise IOError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len == 1:
                        raise IOError(
                            "classlabel tag requires an associated Boolean  " "value"
                        )
                    if tokens[1] == "true":
                        class_labels = True
                    elif tokens[1] == "false":
                        class_labels = False
                    else:
                        raise IOError("invalid classLabel value")
                    # Check if we have any associated class values
                    if token_len == 2 and class_labels:
                        raise IOError(
                            "if the classlabel tag is true then class values "
                            "must be supplied"
                        )
                    has_class_labels_tag = True
                    class_label_list = [token.strip() for token in tokens[2:]]
                    metadata_started = True
                # Check if this line contains the start of data
                elif line.startswith("@data"):
                    if line != "@data":
                        raise IOError("data tag should not have an associated value")
                    if data_started and not metadata_started:
                        raise IOError("metadata must come before data")
                    else:
                        has_data_tag = True
                        data_started = True
                # If the 'data tag has been found then metadata has been
                # parsed and data can be loaded
                elif data_started:
                    # Check that a full set of metadata has been provided
                    if (
                        not has_problem_name_tag
                        or not has_timestamps_tag
                        or not has_univariate_tag
                        or not has_class_labels_tag
                        or not has_data_tag
                    ):
                        raise IOError(
                            "a full set of metadata has not been provided "
                            "before the data"
                        )
                    # Replace any missing values with the value specified
                    line = line.replace("?", replace_missing_vals_with)
                    # Check if we dealing with data that has timestamps
                    if timestamps:
                        # We're dealing with timestamps so cannot just split
                        # line on ':' as timestamps may contain one
                        has_another_value = False
                        has_another_dimension = False
                        timestamp_for_dim = []
                        values_for_dimension = []
                        this_line_num_dim = 0
                        line_len = len(line)
                        char_num = 0
                        while char_num < line_len:
                            # Move through any spaces
                            while char_num < line_len and str.isspace(line[char_num]):
                                char_num += 1
                            # See if there is any more data to read in or if
                            # we should validate that read thus far
                            if char_num < line_len:
                                # See if we have an empty dimension (i.e. no
                                # values)
                                if line[char_num] == ":":
                                    if len(instance_list) < (this_line_num_dim + 1):
                                        instance_list.append([])
                                    instance_list[this_line_num_dim].append(
                                        pd.Series(dtype="object")
                                    )
                                    this_line_num_dim += 1
                                    has_another_value = False
                                    has_another_dimension = True
                                    timestamp_for_dim = []
                                    values_for_dimension = []
                                    char_num += 1
                                else:
                                    # Check if we have reached a class label
                                    if line[char_num] != "(" and class_labels:
                                        class_val = line[char_num:].strip()
                                        if class_val not in class_label_list:
                                            raise IOError(
                                                "the class value '"
                                                + class_val
                                                + "' on line "
                                                + str(line_num + 1)
                                                + " is not "
                                                "valid"
                                            )
                                        class_val_list.append(class_val)
                                        char_num = line_len
                                        has_another_value = False
                                        has_another_dimension = False
                                        timestamp_for_dim = []
                                        values_for_dimension = []
                                    else:
                                        # Read in the data contained within
                                        # the next tuple
                                        if line[char_num] != "(" and not class_labels:
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " does "
                                                "not "
                                                "start "
                                                "with a "
                                                "'('"
                                            )
                                        char_num += 1
                                        tuple_data = ""
                                        while (
                                            char_num < line_len
                                            and line[char_num] != ")"
                                        ):
                                            tuple_data += line[char_num]
                                            char_num += 1
                                        if (
                                            char_num >= line_len
                                            or line[char_num] != ")"
                                        ):
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " does "
                                                "not end"
                                                " with a "
                                                "')'"
                                            )
                                        # Read in any spaces immediately
                                        # after the current tuple
                                        char_num += 1
                                        while char_num < line_len and str.isspace(
                                            line[char_num]
                                        ):
                                            char_num += 1

                                        # Check if there is another value or
                                        # dimension to process after this tuple
                                        if char_num >= line_len:
                                            has_another_value = False
                                            has_another_dimension = False
                                        elif line[char_num] == ",":
                                            has_another_value = True
                                            has_another_dimension = False
                                        elif line[char_num] == ":":
                                            has_another_value = False
                                            has_another_dimension = True
                                        char_num += 1
                                        # Get the numeric value for the
                                        # tuple by reading from the end of
                                        # the tuple data backwards to the
                                        # last comma
                                        last_comma_index = tuple_data.rfind(",")
                                        if last_comma_index == -1:
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains a tuple that has "
                                                "no comma inside of it"
                                            )
                                        try:
                                            value = tuple_data[last_comma_index + 1 :]
                                            value = float(value)
                                        except ValueError:
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains a tuple that does "
                                                "not have a valid numeric "
                                                "value"
                                            )
                                        # Check the type of timestamp that
                                        # we have
                                        timestamp = tuple_data[0:last_comma_index]
                                        try:
                                            timestamp = int(timestamp)
                                            timestamp_is_int = True
                                            timestamp_is_timestamp = False
                                        except ValueError:
                                            timestamp_is_int = False
                                        if not timestamp_is_int:
                                            try:
                                                timestamp = timestamp.strip()
                                                timestamp_is_timestamp = True
                                            except ValueError:
                                                timestamp_is_timestamp = False
                                        # Make sure that the timestamps in
                                        # the file (not just this dimension
                                        # or case) are consistent
                                        if (
                                            not timestamp_is_timestamp
                                            and not timestamp_is_int
                                        ):
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains a tuple that "
                                                "has an invalid timestamp '"
                                                + timestamp
                                                + "'"
                                            )
                                        if (
                                            previous_timestamp_was_int is not None
                                            and previous_timestamp_was_int
                                            and not timestamp_is_int
                                        ):
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains tuples where the "
                                                "timestamp format is "
                                                "inconsistent"
                                            )
                                        if (
                                            prev_timestamp_was_timestamp is not None
                                            and prev_timestamp_was_timestamp
                                            and not timestamp_is_timestamp
                                        ):
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains tuples where the "
                                                "timestamp format is "
                                                "inconsistent"
                                            )
                                        # Store the values
                                        timestamp_for_dim += [timestamp]
                                        values_for_dimension += [value]
                                        #  If this was our first tuple then
                                        #  we store the type of timestamp we
                                        #  had
                                        if (
                                            prev_timestamp_was_timestamp is None
                                            and timestamp_is_timestamp
                                        ):
                                            prev_timestamp_was_timestamp = True
                                            previous_timestamp_was_int = False

                                        if (
                                            previous_timestamp_was_int is None
                                            and timestamp_is_int
                                        ):
                                            prev_timestamp_was_timestamp = False
                                            previous_timestamp_was_int = True
                                        # See if we should add the data for
                                        # this dimension
                                        if not has_another_value:
                                            if len(instance_list) < (
                                                this_line_num_dim + 1
                                            ):
                                                instance_list.append([])

                                            if timestamp_is_timestamp:
                                                timestamp_for_dim = pd.DatetimeIndex(
                                                    timestamp_for_dim
                                                )

                                            instance_list[this_line_num_dim].append(
                                                pd.Series(
                                                    index=timestamp_for_dim,
                                                    data=values_for_dimension,
                                                )
                                            )
                                            this_line_num_dim += 1
                                            timestamp_for_dim = []
                                            values_for_dimension = []
                            elif has_another_value:
                                raise IOError(
                                    "dimension " + str(this_line_num_dim + 1) + " on "
                                    "line "
                                    + str(line_num + 1)
                                    + " ends with a ',' that "
                                    "is not followed by "
                                    "another tuple"
                                )
                            elif has_another_dimension and class_labels:
                                raise IOError(
                                    "dimension " + str(this_line_num_dim + 1) + " on "
                                    "line "
                                    + str(line_num + 1)
                                    + " ends with a ':' while "
                                    "it should list a class "
                                    "value"
                                )
                            elif has_another_dimension and not class_labels:
                                if len(instance_list) < (this_line_num_dim + 1):
                                    instance_list.append([])
                                instance_list[this_line_num_dim].append(
                                    pd.Series(dtype=np.float32)
                                )
                                this_line_num_dim += 1
                                num_dimensions = this_line_num_dim
                            # If this is the 1st line of data we have seen
                            # then note the dimensions
                            if not has_another_value and not has_another_dimension:
                                if num_dimensions is None:
                                    num_dimensions = this_line_num_dim
                                if num_dimensions != this_line_num_dim:
                                    raise IOError(
                                        "line "
                                        + str(line_num + 1)
                                        + " does not have the "
                                        "same number of "
                                        "dimensions as the "
                                        "previous line of "
                                        "data"
                                    )
                        # Check that we are not expecting some more data,
                        # and if not, store that processed above
                        if has_another_value:
                            raise IOError(
                                "dimension "
                                + str(this_line_num_dim + 1)
                                + " on line "
                                + str(line_num + 1)
                                + " ends with a ',' that is "
                                "not followed by another "
                                "tuple"
                            )
                        elif has_another_dimension and class_labels:
                            raise IOError(
                                "dimension "
                                + str(this_line_num_dim + 1)
                                + " on line "
                                + str(line_num + 1)
                                + " ends with a ':' while it "
                                "should list a class value"
                            )
                        elif has_another_dimension and not class_labels:
                            if len(instance_list) < (this_line_num_dim + 1):
                                instance_list.append([])
                            instance_list[this_line_num_dim].append(
                                pd.Series(dtype="object")
                            )
                            this_line_num_dim += 1
                            num_dimensions = this_line_num_dim
                        # If this is the 1st line of data we have seen then
                        # note the dimensions
                        if (
                            not has_another_value
                            and num_dimensions != this_line_num_dim
                        ):
                            raise IOError(
                                "line " + str(line_num + 1) + " does not have the same "
                                "number of dimensions as the "
                                "previous line of data"
                            )
                        # Check if we should have class values, and if so
                        # that they are contained in those listed in the
                        # metadata
                        if class_labels and len(class_val_list) == 0:
                            raise IOError("the cases have no associated class values")
                    else:
                        dimensions = line.split(":")
                        # If first row then note the number of dimensions (
                        # that must be the same for all cases)
                        if is_first_case:
                            num_dimensions = len(dimensions)
                            if class_labels:
                                num_dimensions -= 1
                            for _dim in range(0, num_dimensions):
                                instance_list.append([])
                            is_first_case = False
                        # See how many dimensions that the case whose data
                        # in represented in this line has
                        this_line_num_dim = len(dimensions)
                        if class_labels:
                            this_line_num_dim -= 1
                        # All dimensions should be included for all series,
                        # even if they are empty
                        if this_line_num_dim != num_dimensions:
                            raise IOError(
                                "inconsistent number of dimensions. "
                                "Expecting "
                                + str(num_dimensions)
                                + " but have read "
                                + str(this_line_num_dim)
                            )
                        # Process the data for each dimension
                        for dim in range(0, num_dimensions):
                            dimension = dimensions[dim].strip()

                            if dimension:
                                data_series = dimension.split(",")
                                data_series = [float(i) for i in data_series]
                                instance_list[dim].append(pd.Series(data_series))
                            else:
                                instance_list[dim].append(pd.Series(dtype="object"))
                        if class_labels:
                            class_val_list.append(dimensions[num_dimensions].strip())
            line_num += 1
    # Check that the file was not empty
    if line_num:
        # Check that the file contained both metadata and data
        if metadata_started and not (
            has_problem_name_tag
            and has_timestamps_tag
            and has_univariate_tag
            and has_class_labels_tag
            and has_data_tag
        ):
            raise IOError("metadata incomplete")

        elif metadata_started and not data_started:
            raise IOError("file contained metadata but no data")

        elif metadata_started and data_started and len(instance_list) == 0:
            raise IOError("file contained metadata but no data")
        # Create a DataFrame from the data parsed above
        data = pd.DataFrame(dtype=np.float32)
        for dim in range(0, num_dimensions):
            data["dim_" + str(dim)] = instance_list[dim]
        # Check if we should return any associated class labels separately
        if class_labels:
            if return_separate_X_and_y:
                return data, np.asarray(class_val_list)
            else:
                data["class_vals"] = pd.Series(class_val_list)
                return data
        else:
            return data
    else:
        raise IOError("empty file")



def array_fulfill(init_array, length):
    """array_fulfill Fill the array to the specified length.  

    Fill the array to the specified length. For example, fill the array [1,2,3] to the length 5, the new array is [1,2,3,1,2].

    Args:
        init_array (np.ndarray): the array to be filled.
        length (integer): length to be expanded.

    Returns:
        np.ndarray: the filled array
    """
    assert len(init_array) <= length
    init_array = np.array(init_array)
    if len(init_array) < length:
        ful_array = np.tile(init_array, length//len(init_array))
        tmp = init_array[0:length%(len(init_array))]
        ful_array = np.concatenate((ful_array, tmp), axis=0)
    else:
        ful_array = init_array
    return ful_array


class raw_dataset(Dataset):
    """generated from the origin train/test data, and keep the same partition
    

    this class provides an iterator for splitting the original dataset and includes a number of operation such as shuffle to process the data
    
    train_iter: provide a train data iterator
    test_iter: provide a test data iterator
    shuffle: shuffle the train/test data or both 
    noise_add: add white noise to the data
    train_data_enhencement: data enhencement  
     
    """

    def __init__(self, data_path, divide=None, mode="normal") -> None:
        """__init__ initial function

        init function, load a specific dataset and split it into the train dataset and test dataset.

        Args:
            data_path (string): path to the dataset, absolute and relative paths can be used.
            divide (array, float): The data set is divided into proportions in the order of training, validation and testing.
            mode (str, optional): use the pre-defined train/test dataset or divide the train/test dataset freely. normal: use the pre-defined train/test dataset; free: use the "divide" argument to divide the dataset. Defaults to "normal".

        Raises:
            Exception: _description_
        """
        super().__init__()

        files = os.listdir(path=data_path)
        if len(files) == 4:
            # file is npy format
            pass
        elif len(files) == 2:
            # file is ts format
            if files[0].split("_")[-1].split(".")[0].lower() == "test":
                test = files[0]
                train = files[1]
            else:
                test = files[1]
                train = files[0]
            train_data, train_label = load_from_tsfile(os.path.join(
                data_path, train),
                                                       return_data_type="np3D")
            test_data, test_label = load_from_tsfile(os.path.join(
                data_path, test),
                                                     return_data_type="np3D")
            self.train_path = os.path.join(data_path, train)
            tag  ={ }
            # with open(self.train_path) as f:
            #     tag = _read_header(f, self.train_path)
            if os.path.exists(self.train_path):
                with open(self.train_path, "r") as f:
                    count = 0 
                    for line in f:
                        count += 1
                        if line.startswith("@data") : 
                            break
                        if line.startswith("@"):
                            attrs = line.strip().split(" ")
                            if len(attrs) == 2:
                                tag[attrs[0]] = attrs[1]
                            else:
                                tag[attrs[0]] = attrs[1:-1]
            self.tag = tag
            
            
        elif len(files) == 0:
            raise Exception("No such files, check your path!")

        self.total_data = np.concatenate([train_data, test_data], axis=0)
        self.total_label = np.concatenate([train_label, test_label], axis=0)
        self.total_index = list(range(len(self.total_label)))
        try:
            assert mode in ["free", "normal", "fold"]
        except Exception("unknown mode!") as e:
            print(e)
        
        self.class_num = len(np.unique(train_label))
        self.mode = mode
        # This dict is used for the storage of each class data, if the mode is normal, then only train data would be stored. This dict is designed for the following two reasons: 1.sometimes, we may need all class data appear in one batch. 2.when dividing the data set, train(eval/test) should contain data of all class.
        self.label_index_dict = dict()
        for label in np.unique(train_label):
            self.label_index_dict.update(
                {label: np.argwhere(train_label == label)})
        if mode == "normal":
            self.train_index = list(range(len(train_label)))
            self.test_index = list(
                range(len(train_label), len(self.total_data)))
        elif mode == "free":
            self.train_index = []
            self.test_index = []
            if len(divide) == 3:
                self.eval_index = []
            else:
                assert len(divide) == 2
            for label in np.unique(train_label):
                train_bound = int(divide[0] *
                                  len(self.label_index_dict[label]))
                if train_bound == 0:
                    raise Exception(
                        "train_bound equals 0, try to increase the rate of the train data"
                    )

                self.train_index.append(
                    self.label_index_dict[label][:train_bound])
                if len(divide) == 3:
                    eval_bound = int((divide[0] + divide[1]) *
                                     len(self.label_index_dict[label]))
                    self.eval_index.append(
                        self.label_index_dict[label][train_bound:eval_bound])
                    self.test_index.append(
                        self.label_index_dict[label][eval_bound:])
                    if train_bound == eval_bound:
                        raise Exception(
                            "train_bound equals eval_bound, try to increase the rate of the eval data"
                        )
                    if eval_bound == len(self.label_index_dict[label]):
                        raise Exception(
                            "eval_bound equals whole data length, try to increase the rate of the test data"
                        )
                else:
                    self.test_index.append(
                        self.label_index_dict[label][train_bound:])
                    if train_bound == len(self.label_index_dict[label]):
                        raise Exception(
                            "eval_bound equals whole data length, try to increase the rate of the test data"
                        )
            # convert the label from str into the Long

        elif mode == "fold":
            pass
        self.total_raw_label = self.total_label
        self.total_class = np.unique(train_label)
        self.total_label = label_encoder(self.total_label)
        self.max_length = np.max(np.bincount(self.total_label[self.train_index]))
        for c in self.total_class:
            self.label_index_dict[c] = array_fulfill(self.label_index_dict[c], self.max_length)
        # self.tag = tag

    def test(self, ):
        print(self.class_num)
        print(self.total_label)
        # print(self.test_index)
        # for x,y in self.train_iter(3, data_type="tensor"):
        #     print(x.shape, y)

    def __len__(self, ):
        return len(self.total_index)

    def __getitem__(self, index):
        return self.total_data[index], self.total_label[index]

    def train_iter(self,
                   batch_size,
                   use_last=True,
                   mode="normal",
                   data_type="np3d"):
        """train_iter training data iterator

        _extended_summary_

        Args:
            batch_size (_type_): the batch size of the training data
            use_last (bool, optional): if the whole length is not divisible by the batch_size, then this arguments can choose whether to use the last batch size (Sometimes you may want each batch to be the same size). WARNING: if the number of data left is 1, then batch_norm may cause an Error. Defaults to True.
            mode (str, optional): choose the mode to return the training data.  Can be extended for other task. normal:just return the training data by batch_size. alltype: all class of data will be present in a batch. "batch_size" will be taken as the the number of classes in a training batch, so the actual batch size will be nclass * batch_size. Defaults to "normal".
            data_type (str, optional): the type of the yield data. The optional value: "np3d|np3D" the nparray;"tensor|floattensor" the FloatTensor  Defaults to "np3d". 
        """
        if data_type in ["floattensor", "tensor"]:
            self.total_data = torch.FloatTensor(self.total_data)
            self.total_label = torch.LongTensor(self.total_label)
        
        assert mode in {"normal", "alltype"}
        
        if mode=="normal":            
            for i in range(0, len(self.train_index), batch_size):
                if (i + batch_size) <= len(self.train_index):
                    yield self.total_data[
                        self.train_index[i:i + batch_size]], self.total_label[
                            self.train_index[i:i + batch_size]]
                elif use_last:
                    yield self.total_data[self.train_index[
                        i:len(self.train_index)]], self.total_label[
                            self.train_index[i:len(self.train_index)]]
        elif mode == " alltype":
            for i in range(0, self.max_length, batch_size):
                data = np.array([])
                label = np.array([])
                if (i+batch_size) <= self.max_length:
                    data
                
            

    def test_iter(self, batch_size, use_last, mode="normal"):
        """test_iter testing data iterator

        _extended_summary_

        Args:
            batch_size (_type_): the batch size of the training data
            use_last (bool, optional): if the whole length is not divisible by the batch_size, then this arguments can choose whether to use the last batch size (Sometimes you may want each batch to be the same size). WARNING: if the number of data left is 1, then batch_norm may cause an Error. Defaults to True.
            mode (str, optional): choose the mode to return the training data.  Can be extended for other task. Defaults to "normal".
        Yields:
            X_data: data
            Y_Label: label
        """
        if mode == " normal":

            for i in range(0, len(self.test_index), batch_size):
                if (i + batch_size) <= len(self.test_index):
                    yield self.total_data[
                        self.test_index[i:i + batch_size]], self.total_label[
                            self.test_index[i:i + batch_size]]
                elif use_last:
                    yield self.total_data[self.test_index[
                        i:len(self.test_index)]], self.total_label[
                            self.test_index[i:len(self.test_index)]]

    def shuffle(self, ):
        """shuffle shuffle the train/test index

        To make data different in each batch, add randomness 
        """
        np.random.shuffle(self.train_index)
        np.random.shuffle(self.test_index)
        for i in self.total_class:
            np.random.shuffle(self.label_index_dict[i])

    def read_tag(self, ):
        """read_tag _summary_

        _extended_summary_

        Returns:
            Dict: the tags of the dataset
        """
        from sktime.datasets._data_io import _read_header
        
        
        return self.tag

    def noise_add():
        pass

    def train_data_enhencement():
        pass


def label_encoder(label):
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    return np.array(label_encoder.fit_transform(label))



def load_raw_ts(dataset, tensor_format=True, path='data/'):
    path = "{}raw/{}/".format(path, dataset)
    # path = path + "Multivariate_ts" + "/"
    # if it is npy file 
    if os.path.exists(path+'X_train.npy'):
        x_train = np.load(path + 'X_train.npy')
        y_train = np.load(path + 'y_train.npy')
        x_test = np.load(path + 'X_test.npy')
        y_test = np.load(path + 'y_test.npy')
        ts = np.concatenate((x_train, x_test), axis=0)
        ts = np.transpose(ts, axes=(0, 2, 1))
        labels = np.concatenate((y_train, y_test), axis=0)
        nclass = int(np.amax(labels)) + 1


        train_size = y_train.shape[0]

        total_size = labels.shape[0]
        idx_train = range(train_size)
        idx_val = range(train_size, total_size)
        idx_test = range(train_size, total_size)
    # if it is ts file 
    elif os.path.exists(path + dataset + "_TEST.ts"):
        test_X, test_y = load_from_tsfile2dataframe(path+dataset+'_TEST.ts')
        train_X, train_y = load_from_tsfile2dataframe(path+dataset+'_TRAIN.ts')
        label_encoder = LabelEncoder()
        labels = np.concatenate((np.array(label_encoder.fit_transform(train_y)),np.array(label_encoder.fit_transform(test_y))),axis=0)
        nclass = int(np.amax(labels)) + 1 
        train_nd = series_to_nd(train_X.values)
        test_nd = series_to_nd(test_X.values)
        ts = np.concatenate((train_nd,test_nd), axis=0)
        np.transpose(ts, (0, 2, 1))
        train_size = train_y.shape[0]
        total_size = labels.shape[0]
        idx_train = range(train_size)
        idx_val = range(train_size, total_size)
        idx_test = range(train_size, total_size)
    if tensor_format:
        # features = torch.FloatTensor(np.array(features))
        ts = torch.FloatTensor(np.array(ts))
        labels = torch.LongTensor(labels)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

    return ts, labels, idx_train, idx_val, idx_test, nclass


   
#-----------------------------------------------------------#

#   Data iterator for UAE archive

#-----------------------------------------------------------#
# todo 增加噪声参数

class Data_iter():
    '''data iterator for the UAE Archive

    
    '''
    def __init__(self, dataset, division) -> None:
        self.ts, self.labels, _,_,_,_ = load_raw_ts(dataset, tensor_format=False)
        assert len(division)==3
        self.division = np.array(division)/np.sum(division)
        self.classes = np.unique(self.labels)
        self.class_dict = {}
        self.max_length = np.max(np.bincount(self.labels))
        self.min_length = np.min(np.bincount(self.labels))
        for i in self.classes:
            self.class_dict[i] = [j for j,x in enumerate(self.labels) if x==i]
        if self.min_length * division[1] < 1:
            raise
    

        
    def train_iter(self, batch_sample_size, use_last_data=True, tensor_format=True, use_noise=False):
        r'''Training Data Iterator

        This function was give a training data iterator which is divided by the division. The same size of every class train data are given in each batch.

        Args:
          batch_sample_size: interval, the amount of the single class data in a batch
          use_last_data: default True, whether use the last data which is not same size of the batch_sample_size
          tensor_format: default True, whether return the torch.Tensor or numpy.ndarray
          use_noise: default False, whether use the noise added to the time series 

        '''
        tmp_class_idx = {}
        tmp_maxlength = int(self.max_length*self.division[0])
        for i in self.classes:
            tmp_class_idx[i] = array_fulfill(self.class_dict[i][0:int(self.division[0]*len(self.class_dict[i]))], tmp_maxlength)    # 填充每一类的数量为最大值
        
        for i in range(tmp_maxlength//batch_sample_size):
            tmp_batch_idx = []
            for j in self.classes:
                tmp_batch_idx.append(tmp_class_idx[j][batch_sample_size*i: batch_sample_size*(i+1)])
            tmp_batch_idx = np.concatenate(tmp_batch_idx, axis=0)

            ts = self.ts[tmp_batch_idx] 
            if use_noise:
                ts += np.random.normal(0, 0.01, [ts.shape[1], ts.shape[2]])
            
            if tensor_format:
                yield torch.FloatTensor(ts), torch.LongTensor(self.labels[tmp_batch_idx])
            else:
                yield ts, self.labels[tmp_batch_idx]  


        if use_last_data and (tmp_maxlength%batch_sample_size is not 0):
            tmp_batch_idx = []
            for j in self.classes:
                tmp_batch_idx.append(tmp_class_idx[j][-(tmp_maxlength%batch_sample_size)+1:])
            tmp_batch_idx = np.concatenate(tmp_batch_idx, axis=0)
            ts = self.ts[tmp_batch_idx]
            if use_noise:
                ts += np.random.normal(0, 0.01, [ts.shape[1], ts.shape[2]])
            if tensor_format:
                yield torch.FloatTensor(self.ts[tmp_batch_idx]), torch.LongTensor(self.labels[tmp_batch_idx])
            else:
                yield self.ts[tmp_batch_idx], self.labels[tmp_batch_idx]  

    def eval_iter(self, batch_sample_size, tensor_format=True): 
        tmp_class_idx = {}
        tmp_maxlength = int(self.max_length*self.division[1])
        for i in self.classes:
            tmp_class_idx[i] = array_fulfill(self.class_dict[i][int(self.division[0]*len(self.class_dict[i])):int((self.division[1]+self.division[0])*len(self.class_dict[i]))], tmp_maxlength)    # 填充每一类的数量为最大值
        
        for i in range(tmp_maxlength//batch_sample_size):
            tmp_batch_idx = []
            for j in self.classes:
                tmp_batch_idx.append(tmp_class_idx[j][batch_sample_size*i: batch_sample_size*(i+1)])
            tmp_batch_idx = np.concatenate(tmp_batch_idx, axis=0)
            if tensor_format:
                yield torch.FloatTensor(self.ts[tmp_batch_idx]), torch.LongTensor(self.labels[tmp_batch_idx])
            else:
                yield self.ts[tmp_batch_idx], self.labels[tmp_batch_idx]  


        tmp_batch_idx = []
        for j in self.classes:
            tmp_batch_idx.append(tmp_class_idx[j][-(self.max_length%batch_sample_size)+1:])
        tmp_batch_idx = np.concatenate(tmp_batch_idx, axis=0)
        if tensor_format:
            yield torch.FloatTensor(self.ts[tmp_batch_idx]), torch.LongTensor(self.labels[tmp_batch_idx])
        else:
            yield self.ts[tmp_batch_idx], self.labels[tmp_batch_idx] 

    def test_iter(self, batch_sample_size, tensor_format=True):
        tmp_class_idx = {}
        tmp_maxlength = int(self.max_length*self.division[2])
        for i in self.classes:
            tmp_class_idx[i] = array_fulfill(self.class_dict[i][int(self.division[1]*len(self.class_dict[i])):], tmp_maxlength)    # 填充每一类的数量为最大值
        
        for i in range(tmp_maxlength//batch_sample_size):
            tmp_batch_idx = []
            for j in self.classes:
                tmp_batch_idx.append(tmp_class_idx[j][batch_sample_size*i: batch_sample_size*(i+1)])
            tmp_batch_idx = np.concatenate(tmp_batch_idx, axis=0)
            if tensor_format:
                yield torch.FloatTensor(self.ts[tmp_batch_idx]), torch.LongTensor(self.labels[tmp_batch_idx])
            else:
                yield self.ts[tmp_batch_idx], self.labels[tmp_batch_idx]  


        tmp_batch_idx = []
        for j in self.classes:
            tmp_batch_idx.append(tmp_class_idx[j][-(self.max_length%batch_sample_size)+1:])
        tmp_batch_idx = np.concatenate(tmp_batch_idx, axis=0)
        if tensor_format:
            yield torch.FloatTensor(self.ts[tmp_batch_idx]), torch.LongTensor(self.labels[tmp_batch_idx])
        else:
            yield self.ts[tmp_batch_idx], self.labels[tmp_batch_idx] 


    def shuffle(self, ) -> None:
        for i in self.classes:
            np.random.shufffle(self.class_dict[i])
    
    def data_shape(self):
        return self.ts.shape

    def label_num(self):
        return len(self.classes)


def ts_equal_length(path):
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().lower()
            if line:
                if line.startswith("@equallength") :
                    tokens = line.split(" ")
                    if tokens[1] == "false":
                        return False
                    elif tokens[1] == "true":
                        return True
    return True
    raise Exception("No equallength")
     
#-----------------------------------------------------------#

#   用于获取当前时间，保存数据时可以作为标签

#-----------------------------------------------------------#

def get_time_str(style='Nonetype'):
    t = time.localtime()
    if style is 'Nonetype':
        return ("{}{}{}{}{}{}".format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec))
    elif style is 'underline':
        return ("{}_{}_{}_{}_{}_{}".format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec))
    
def reg_find_index(str_array, reg_str):
    """This func is used to find the the elements of the initial character in the array 

    This func is designed worked as the docker Id, for example:
    if the array is ["ab", "abc", "cba"]
    Then the initial chars are "ab", we hope to get the "ab" and the "abc"
    Then the initial chars are "c", we hope to get the "cba" 
    
    Args:
        str_array (_type_): the string list 
        reg_str (_type_): the initial characters 

    Returns:
        List: the indexs of the matched strings
    """
    
    # generate the regular expression "^($string)."
    reg_str = "^({}).".format(reg_str.lower())
    
    indexs = []
    for i in range(len(str_array)):
        str_ = str_array[i].lower()
        if re.match(reg_str, str_):
            indexs.append(i)
    return indexs 



        
         


def load_from_tsfile2nparray(
    full_file_path_and_name,
    return_separate_X_and_y=True,
    replace_missing_vals_with="NaN",
):
    """Load data from a .ts file into a Pandas DataFrame.

    Parameters
    ----------
    full_file_path_and_name: str
        The full pathname of the .ts file to read.
    return_separate_X_and_y: bool
        true if X and Y values should be returned as separate Data Frames (
        X) and a numpy array (y), false otherwise.
        This is only relevant for data that
    replace_missing_vals_with: str
       The value that missing values in the text file should be replaced
       with prior to parsing.

    Returns
    -------
    DataFrame (default) or ndarray (i
        If return_separate_X_and_y then a tuple containing a DataFrame and a
        numpy array containing the relevant time-series and corresponding
        class values.
    DataFrame
        If not return_separate_X_and_y then a single DataFrame containing
        all time-series and (if relevant) a column "class_vals" the
        associated class values.
    """
    # Initialize flags and variables used when parsing the file
    metadata_started = False
    data_started = False

    has_problem_name_tag = False
    has_timestamps_tag = False
    has_univariate_tag = False
    has_class_labels_tag = False
    has_data_tag = False

    previous_timestamp_was_int = None
    prev_timestamp_was_timestamp = None
    num_dimensions = None
    is_first_case = True
    instance_list = []
    class_val_list = []
    line_num = 0
    datas_tuple = []
    # Parse the file
    with open(full_file_path_and_name, "r", encoding="utf-8") as file:
        for line in file:
            # Strip white space from start/end of line and change to
            # lowercase for use below
            line = line.strip().lower()
            # Empty lines are valid at any point in a file
            if line:
                # Check if this line contains metadata
                # Please note that even though metadata is stored in this
                # function it is not currently published externally
                if line.startswith("@problemname"):
                    # Check that the data has not started
                    if data_started:
                        raise IOError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len == 1:
                        raise IOError("problemname tag requires an associated value")
                    # problem_name = line[len("@problemname") + 1:]
                    has_problem_name_tag = True
                    metadata_started = True
                elif line.startswith("@timestamps"):
                    # Check that the data has not started
                    if data_started:
                        raise IOError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len != 2:
                        raise IOError(
                            "timestamps tag requires an associated Boolean " "value"
                        )
                    elif tokens[1] == "true":
                        timestamps = True
                    elif tokens[1] == "false":
                        timestamps = False
                    else:
                        raise IOError("invalid timestamps value")
                    has_timestamps_tag = True
                    metadata_started = True
                elif line.startswith("@univariate"):
                    # Check that the data has not started
                    if data_started:
                        raise IOError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len != 2:
                        raise IOError(
                            "univariate tag requires an associated Boolean  " "value"
                        )
                    elif tokens[1] == "true":
                        # univariate = True
                        pass
                    elif tokens[1] == "false":
                        # univariate = False
                        pass
                    else:
                        raise IOError("invalid univariate value")
                    has_univariate_tag = True
                    metadata_started = True
                elif line.startswith("@classlabel"):
                    # Check that the data has not started
                    if data_started:
                        raise IOError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len == 1:
                        raise IOError(
                            "classlabel tag requires an associated Boolean  " "value"
                        )
                    if tokens[1] == "true":
                        class_labels = True
                    elif tokens[1] == "false":
                        class_labels = False
                    else:
                        raise IOError("invalid classLabel value")
                    # Check if we have any associated class values
                    if token_len == 2 and class_labels:
                        raise IOError(
                            "if the classlabel tag is true then class values "
                            "must be supplied"
                        )
                    has_class_labels_tag = True
                    class_label_list = [token.strip() for token in tokens[2:]]
                    metadata_started = True
                # Check if this line contains the start of data
                elif line.startswith("@data"):
                    if line != "@data":
                        raise IOError("data tag should not have an associated value")
                    if data_started and not metadata_started:
                        raise IOError("metadata must come before data")
                    else:
                        has_data_tag = True
                        data_started = True
                # If the 'data tag has been found then metadata has been
                # parsed and data can be loaded
                elif data_started:
                    # Check that a full set of metadata has been provided
                    if (
                        not has_problem_name_tag
                        or not has_timestamps_tag
                        or not has_univariate_tag
                        or not has_class_labels_tag
                        or not has_data_tag
                    ):
                        raise IOError(
                            "a full set of metadata has not been provided "
                            "before the data"
                        )
                    # Replace any missing values with the value specified
                    line = line.replace("?", replace_missing_vals_with)
                    # Check if we dealing with data that has timestamps
                    if timestamps:
                        # We're dealing with timestamps so cannot just split
                        # line on ':' as timestamps may contain one
                        has_another_value = False
                        has_another_dimension = False
                        timestamp_for_dim = []
                        values_for_dimension = []
                        this_line_num_dim = 0
                        line_len = len(line)
                        char_num = 0
                        while char_num < line_len:
                            # Move through any spaces
                            while char_num < line_len and str.isspace(line[char_num]):
                                char_num += 1
                            # See if there is any more data to read in or if
                            # we should validate that read thus far
                            if char_num < line_len:
                                # See if we have an empty dimension (i.e. no
                                # values)
                                if line[char_num] == ":":
                                    if len(instance_list) < (this_line_num_dim + 1):
                                        instance_list.append([])
                                    instance_list[this_line_num_dim].append(
                                        pd.Series(dtype="object")
                                    )
                                    this_line_num_dim += 1
                                    has_another_value = False
                                    has_another_dimension = True
                                    timestamp_for_dim = []
                                    values_for_dimension = []
                                    char_num += 1
                                else:
                                    # Check if we have reached a class label
                                    if line[char_num] != "(" and class_labels:
                                        class_val = line[char_num:].strip()
                                        if class_val not in class_label_list:
                                            raise IOError(
                                                "the class value '"
                                                + class_val
                                                + "' on line "
                                                + str(line_num + 1)
                                                + " is not "
                                                "valid"
                                            )
                                        class_val_list.append(class_val)
                                        char_num = line_len
                                        has_another_value = False
                                        has_another_dimension = False
                                        timestamp_for_dim = []
                                        values_for_dimension = []
                                    else:
                                        # Read in the data contained within
                                        # the next tuple
                                        if line[char_num] != "(" and not class_labels:
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " does "
                                                "not "
                                                "start "
                                                "with a "
                                                "'('"
                                            )
                                        char_num += 1
                                        tuple_data = ""
                                        while (
                                            char_num < line_len
                                            and line[char_num] != ")"
                                        ):
                                            tuple_data += line[char_num]
                                            char_num += 1
                                        if (
                                            char_num >= line_len
                                            or line[char_num] != ")"
                                        ):
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " does "
                                                "not end"
                                                " with a "
                                                "')'"
                                            )
                                        # Read in any spaces immediately
                                        # after the current tuple
                                        char_num += 1
                                        while char_num < line_len and str.isspace(
                                            line[char_num]
                                        ):
                                            char_num += 1

                                        # Check if there is another value or
                                        # dimension to process after this tuple
                                        if char_num >= line_len:
                                            has_another_value = False
                                            has_another_dimension = False
                                        elif line[char_num] == ",":
                                            has_another_value = True
                                            has_another_dimension = False
                                        elif line[char_num] == ":":
                                            has_another_value = False
                                            has_another_dimension = True
                                        char_num += 1
                                        # Get the numeric value for the
                                        # tuple by reading from the end of
                                        # the tuple data backwards to the
                                        # last comma
                                        last_comma_index = tuple_data.rfind(",")
                                        if last_comma_index == -1:
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains a tuple that has "
                                                "no comma inside of it"
                                            )
                                        try:
                                            value = tuple_data[last_comma_index + 1 :]
                                            value = float(value)
                                        except ValueError:
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains a tuple that does "
                                                "not have a valid numeric "
                                                "value"
                                            )
                                        # Check the type of timestamp that
                                        # we have
                                        timestamp = tuple_data[0:last_comma_index]
                                        try:
                                            timestamp = int(timestamp)
                                            timestamp_is_int = True
                                            timestamp_is_timestamp = False
                                        except ValueError:
                                            timestamp_is_int = False
                                        if not timestamp_is_int:
                                            try:
                                                timestamp = timestamp.strip()
                                                timestamp_is_timestamp = True
                                            except ValueError:
                                                timestamp_is_timestamp = False
                                        # Make sure that the timestamps in
                                        # the file (not just this dimension
                                        # or case) are consistent
                                        if (
                                            not timestamp_is_timestamp
                                            and not timestamp_is_int
                                        ):
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains a tuple that "
                                                "has an invalid timestamp '"
                                                + timestamp
                                                + "'"
                                            )
                                        if (
                                            previous_timestamp_was_int is not None
                                            and previous_timestamp_was_int
                                            and not timestamp_is_int
                                        ):
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains tuples where the "
                                                "timestamp format is "
                                                "inconsistent"
                                            )
                                        if (
                                            prev_timestamp_was_timestamp is not None
                                            and prev_timestamp_was_timestamp
                                            and not timestamp_is_timestamp
                                        ):
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains tuples where the "
                                                "timestamp format is "
                                                "inconsistent"
                                            )
                                        # Store the values
                                        timestamp_for_dim += [timestamp]
                                        values_for_dimension += [value]
                                        #  If this was our first tuple then
                                        #  we store the type of timestamp we
                                        #  had
                                        if (
                                            prev_timestamp_was_timestamp is None
                                            and timestamp_is_timestamp
                                        ):
                                            prev_timestamp_was_timestamp = True
                                            previous_timestamp_was_int = False

                                        if (
                                            previous_timestamp_was_int is None
                                            and timestamp_is_int
                                        ):
                                            prev_timestamp_was_timestamp = False
                                            previous_timestamp_was_int = True
                                        # See if we should add the data for
                                        # this dimension
                                        if not has_another_value:
                                            if len(instance_list) < (
                                                this_line_num_dim + 1
                                            ):
                                                instance_list.append([])

                                            if timestamp_is_timestamp:
                                                timestamp_for_dim = pd.DatetimeIndex(
                                                    timestamp_for_dim
                                                )

                                            instance_list[this_line_num_dim].append(
                                                pd.Series(
                                                    index=timestamp_for_dim,
                                                    data=values_for_dimension,
                                                )
                                            )
                                            this_line_num_dim += 1
                                            timestamp_for_dim = []
                                            values_for_dimension = []
                            elif has_another_value:
                                raise IOError(
                                    "dimension " + str(this_line_num_dim + 1) + " on "
                                    "line "
                                    + str(line_num + 1)
                                    + " ends with a ',' that "
                                    "is not followed by "
                                    "another tuple"
                                )
                            elif has_another_dimension and class_labels:
                                raise IOError(
                                    "dimension " + str(this_line_num_dim + 1) + " on "
                                    "line "
                                    + str(line_num + 1)
                                    + " ends with a ':' while "
                                    "it should list a class "
                                    "value"
                                )
                            elif has_another_dimension and not class_labels:
                                if len(instance_list) < (this_line_num_dim + 1):
                                    instance_list.append([])
                                instance_list[this_line_num_dim].append(
                                    pd.Series(dtype=np.float32)
                                )
                                this_line_num_dim += 1
                                num_dimensions = this_line_num_dim
                            # If this is the 1st line of data we have seen
                            # then note the dimensions
                            if not has_another_value and not has_another_dimension:
                                if num_dimensions is None:
                                    num_dimensions = this_line_num_dim
                                if num_dimensions != this_line_num_dim:
                                    raise IOError(
                                        "line "
                                        + str(line_num + 1)
                                        + " does not have the "
                                        "same number of "
                                        "dimensions as the "
                                        "previous line of "
                                        "data"
                                    )
                        # Check that we are not expecting some more data,
                        # and if not, store that processed above
                        if has_another_value:
                            raise IOError(
                                "dimension "
                                + str(this_line_num_dim + 1)
                                + " on line "
                                + str(line_num + 1)
                                + " ends with a ',' that is "
                                "not followed by another "
                                "tuple"
                            )
                        elif has_another_dimension and class_labels:
                            raise IOError(
                                "dimension "
                                + str(this_line_num_dim + 1)
                                + " on line "
                                + str(line_num + 1)
                                + " ends with a ':' while it "
                                "should list a class value"
                            )
                        elif has_another_dimension and not class_labels:
                            if len(instance_list) < (this_line_num_dim + 1):
                                instance_list.append([])
                            instance_list[this_line_num_dim].append(
                                pd.Series(dtype="object")
                            )
                            this_line_num_dim += 1
                            num_dimensions = this_line_num_dim
                        # If this is the 1st line of data we have seen then
                        # note the dimensions
                        if (
                            not has_another_value
                            and num_dimensions != this_line_num_dim
                        ):
                            raise IOError(
                                "line " + str(line_num + 1) + " does not have the same "
                                "number of dimensions as the "
                                "previous line of data"
                            )
                        # Check if we should have class values, and if so
                        # that they are contained in those listed in the
                        # metadata
                        if class_labels and len(class_val_list) == 0:
                            raise IOError("the cases have no associated class values")
                    else:
                        dimensions = line.split(":")
                        # If first row then note the number of dimensions (
                        # that must be the same for all cases)
                        if is_first_case:
                            num_dimensions = len(dimensions)
                            if class_labels:
                                num_dimensions -= 1
                            for _dim in range(0, num_dimensions):
                                instance_list.append([])
                            is_first_case = False
                        # See how many dimensions that the case whose data
                        # in represented in this line has
                        this_line_num_dim = len(dimensions)
                        if class_labels:
                            this_line_num_dim -= 1
                        # All dimensions should be included for all series,
                        # even if they are empty
                        if this_line_num_dim != num_dimensions:
                            raise IOError(
                                "inconsistent number of dimensions. "
                                "Expecting "
                                + str(num_dimensions)
                                + " but have read "
                                + str(this_line_num_dim)
                            )
                        data_tuple = []
                        # Process the data for each dimension
                        for dim in range(0, num_dimensions):
                            dimension = dimensions[dim].strip()

                            if dimension:
                                data_series = dimension.split(",")
                                data_series = [float(i) for i in data_series]
                                data_tuple.append(np.array(data_series))
                                # instance_list[dim].append(np.array(data_series))
                            else:
                                data_tuple.append(np.array([],dtype="object"))
                                # instance_list[dim].append(np.array(dtype="object"))
                                
                        if class_labels:
                            class_val_list.append(dimensions[num_dimensions].strip())
                        datas_tuple.append(np.array(data_tuple, dtype="float32"))
            line_num += 1
    # Check that the file was not empty
    if line_num:
        # Check that the file contained both metadata and data
        if metadata_started and not (
            has_problem_name_tag
            and has_timestamps_tag
            and has_univariate_tag
            and has_class_labels_tag
            and has_data_tag
        ):
            raise IOError("metadata incomplete")

        elif metadata_started and not data_started:
            raise IOError("file contained metadata but no data")

        elif metadata_started and data_started and len(instance_list) == 0:
            raise IOError("file contained metadata but no data")
        # Create a DataFrame from the data parsed above
        data = np.array(datas_tuple)
            
        # Check if we should return any associated class labels separately
        if class_labels:
            if return_separate_X_and_y:
                return data, np.asarray(class_val_list)
            else:
                data["class_vals"] = np.array(class_val_list)
                return data
        else:
            return data
    else:
        raise IOError("empty file")
    
    
    
   
def load_data(dataset, train=True):
    path = [
        './',
        './data/',
        '../data/',
        '../',
        './data/raw/',
    ]
    for p in path:
        if os.path.exists(p + dataset):
            break
    if train:
        dataset_path = os.path.join(p, dataset, f'{dataset}_TRAIN.ts')
    else:
        dataset_path = os.path.join(p, dataset, f'{dataset}_TEST.ts')
    
    return load_from_tsfile2nparray(dataset_path) 
    
    
class UEADataset(Dataset):
    def __init__(self, dataname, train=True) -> None:
        super().__init__()
        self.data, self.label = load_data(dataname, train=train)
        
        self.idx = np.arange(len(self.data))
        
    def __getitem__(self, index):    
        return self.data[index], self.label[index]
        
    def __len__(self):
        return len(self.data)
    
class MyOnehot():
    def __init__(self, labels):
        self.labels = np.unique(labels)
        self.onehot_matrix = np.eye(len(self.labels))
        
    def transform(self, input):
        # print(self.labels.shape, input.shape)
        return (input==self.labels.reshape(-1,1)).astype(int).T