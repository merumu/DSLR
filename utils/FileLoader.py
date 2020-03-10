import pandas

class FileLoader:
    def __init__(self):
        pass

    def load(self, path):
        try:
            csv_file = pandas.read_csv(path)
            df = pandas.DataFrame(csv_file)
            print("Loading dataset of dimensions {} x {}".format(len(df.axes[0]), len(df.axes[1])))
            return df
        except:
            print("error: incorrect path or incorrect csv file")
            exit()