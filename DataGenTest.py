from DataGenerator.DataGenerator import TrajDataset, ArrangeData, ProperState, DataGenerator
import warnings

def main():
    warnings.filterwarnings("ignore")

    gen = DataGenerator(num_steps=20, num_init=2, controls=[1])
    generated_data = gen.generate()

    trajs = TrajDataset(generated_data, num_steps=4)

    print(generated_data['x'][0:1].shape)
    print(generated_data['u'][0:1])
    print(generated_data['t'][0:1])



if __name__ == "__main__":
    main()
