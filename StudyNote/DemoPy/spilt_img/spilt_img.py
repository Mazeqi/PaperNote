import os
import shutil


class Preprocessor:
    def __init__(self):
        self.data_dir = 'cap2020.10'

        # 01, 02, 03, 04, 05, 06
        self.label = ['Top', 'Front', 'Rear', 'Right', 'Left', 'Bottom']

        # 复制到哪个文件夹去
        self.cp_dir = '2020.10'

        self.reorg_train_valid()

    def mkdir_if_not_exist(self, path):
        if not os.path.exists(os.path.join(*path)):
            os.makedirs(os.path.join(*path))


    def reorg_train_valid(self):

        for train_file in os.listdir(self.data_dir):

            idx = int(train_file.split('.')[0][-1]) - 1
            label = self.label[idx]

            self.mkdir_if_not_exist([self.cp_dir, label])
            shutil.copy(os.path.join(self.data_dir, train_file),
                        os.path.join(self.cp_dir, label)
                        )

if __name__ == '__main__':
    test = Preprocessor()
    # test.reorg_train_valid()

