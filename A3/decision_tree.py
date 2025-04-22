import sys
import os
import time

from dt.parta import parta
from dt.partb import partb
from dt.partc import partc
from dt.partd_i import partdi
from dt.parte import parte

def main():

    train_data_path = sys.argv[1]
    validation_data_path = sys.argv[2]
    test_data_path = sys.argv[3]
    output_folder_path = sys.argv[4]
    question_part = sys.argv[5].lower()

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)


    if question_part == 'a':
        parta(train_data_path, validation_data_path, test_data_path, output_folder_path)
    elif question_part == 'b':
        partb(train_data_path, validation_data_path, test_data_path, output_folder_path)
    elif question_part == 'c':
        partc(train_data_path, validation_data_path, test_data_path, output_folder_path)
    elif question_part == 'd':
        partdi(train_data_path, validation_data_path, test_data_path, output_folder_path)
    elif question_part == 'e':
        parte(train_data_path, validation_data_path, test_data_path, output_folder_path)

if __name__ == "__main__":
    main()
