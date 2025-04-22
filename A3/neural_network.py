import sys
import os
import time

from nn.partb import partb
from nn.partc import partc
from nn.partd import partd
from nn.parte import parte
from nn.partf import partf

def main():

    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    output_folder_path = sys.argv[3]
    question_part = sys.argv[4].lower()

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    output_file=output_folder_path+f"/prediction_{question_part}.csv"
    if question_part == 'b':
        partb(train_data_path, test_data_path, output_file)
    elif question_part == 'c':
        partc(train_data_path, test_data_path, output_file)
    elif question_part == 'd':
        partd(train_data_path, test_data_path, output_file)
    elif question_part == 'e':
        parte(train_data_path, test_data_path, output_file)
    elif question_part == 'f':
        partf(train_data_path, test_data_path, output_file)

if __name__ == "__main__":
    main()
