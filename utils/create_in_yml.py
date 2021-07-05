import yaml
import itertools
import argparse
import random
import sys

def eprint(args):
    sys.stderr.write(str(args) + "\n")

MAX_CONF = 512

# Parse arguments
parser = argparse.ArgumentParser(
    description='Create input YAML file for image processing.')

parser.add_argument('--conf', type=int, metavar="INT",
                    help="Number of output configurations.")

# parser.add_argument('--data_folder', metavar="STRING", nargs='*', help="data_folder")

parser.add_argument('--model_name', metavar='STRING', nargs='*',
                    help="model_name", default=['base']) 

parser.add_argument('--model_type', metavar='STRING', nargs='*',
                    help="model_type", default=['unet']) 

parser.add_argument('--conf_type', metavar='STRING', nargs='*',
                    help="conf_type", default=['train']) 

parser.add_argument('--batch_size', type=int, metavar="INT", nargs='*',
                    help="Number of images in a batch to process together in memory.", default=[4])

parser.add_argument('--epoch', type=int, metavar='INT', nargs='*',
                    help="epoch", default=[100])

parser.add_argument('--model_no', type=int, metavar='INT', nargs='*',
                    help="model_no", default=[1])

parser.add_argument('--gpu_no', metavar="STRING", nargs='*', 
                    help="gpu_no", default=['0'])

parser.add_argument('--learning_rate', type=float, metavar='FLOAT', nargs='*',
                    help="Set learning rate for gradient descent", default=[0.001])


parser.add_argument('--size_img', type=int, metavar='INT', nargs='*',
                    help="size_img", default=[512])

parser.add_argument('--scale_factor', type=int, metavar='INT', nargs='*',
                    help="scale_factor", default=[1])

parser.add_argument('--thold_tbud', type=int, metavar='INT', nargs='*',
                    help="thold_tbud", default=[0])
 

parser.add_argument('--dropout_ratio', type=float, metavar='FLOAT', nargs='*',
                    help="dropout_ratio", default=[0.2])

parser.add_argument('--dir_write',  metavar="STRING", nargs='*', 
                    help="dir_write", default=["outputs/training"])

args = parser.parse_args()
filtered_args = {k: v for k, v in vars(args).items() if v is not None and k != 'conf'}

def product_dict(**kwargs):
    keys = kwargs.keys()
    print(keys)
    vals = kwargs.values()
    print(vals)
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

configurations = list(product_dict(**filtered_args))

if args.conf:
    if args.conf <= len(configurations):
        number_of_configurations = args.conf
        eprint("Number of selected configurations is {}.".format(number_of_configurations))
    else:
        number_of_configurations = len(configurations)
        eprint( "WARNING: Selected number of configurations higher then the total number! Maximum available number is {}.".format(number_of_configurations))

else:
    number_of_configurations = len(configurations)
    eprint("Number of available configurations is {}.".format(number_of_configurations))

if number_of_configurations > MAX_CONF:
    eprint( "WARNING: Number of configurations higher than defined maximum ({}).".format(MAX_CONF))
    number_of_configurations = MAX_CONF

eprint('Outputting {} configuration/s:'.format(str(number_of_configurations)))
count = 0
used_random = []
while count < number_of_configurations:
    random_int = random.randint(0, len(configurations) - 1)
    while random_int in used_random:
        random_int = random.randint(0, len(configurations) - 1)
    data = configurations[random_int]
    used_random.append(random_int)
    eprint(str(count+1) + ': ' + str(data))
    with open('inputs_' + str(count) + '.yml', 'w') as outfile:
        yaml.dump(data, outfile)
    count += 1
