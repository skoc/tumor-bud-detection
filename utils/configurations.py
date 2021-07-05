import yaml
import io


class Configurations:
    def __init__(self, file_parameters):
        with io.open(file_parameters) as f:
            parameters_dict = yaml.load(f, Loader=yaml.FullLoader)

        if parameters_dict["conf_type"] == 'train':

            # self.data_folder = parameters_dict["data_folder"]
            self.batch_size = parameters_dict["batch_size"]
            self.epoch = parameters_dict["epoch"]
            self.model_no = parameters_dict["model_no"]
            self.activation = parameters_dict["activation"]
            self.gpu_no = parameters_dict["gpu_no"]
            self.model_name = parameters_dict["model_name"]
            self.model_type = parameters_dict["model_type"]
            self.learning_rate = parameters_dict["learning_rate"]
            self.size_img = parameters_dict["size_img"]
            self.scale_factor = parameters_dict["scale_factor"]
            self.dir_write = parameters_dict["dir_write"]
            self.thold_tbud = parameters_dict["thold_tbud"]
            self.dropout_ratio = parameters_dict["dropout_ratio"]
            self.dropout_level = parameters_dict["dropout_level"]
            self.model_string = "_".join([str(key)+"-"+str(val) for key, val in parameters_dict.items() if key not in ["data_folder", "dir_write"]])
        
        elif parameters_dict["conf_type"] == 'test':

            self.data_folder = parameters_dict["data_folder"]
            self.model_name = parameters_dict["model_name"]
            self.sample_count = parameters_dict["sample_count"]
            self.size_img = parameters_dict["size_img"]
            self.output_folder = parameters_dict["output_folder"]
            self.trained_model = parameters_dict["trained_model"]
            self.gpu_no = parameters_dict["gpu_no"]
            self.model_string = "_".join([str(key)+"-"+str(val) for key, val in parameters_dict.items() if key not in ["data_folder", "dir_write"]])