import argparse
import os
import uuid
from dataset import loadImgData, loadData, saveData
from eval import train, saveModel

def main():
    parser = argparse.ArgumentParser(description='train a model with your params')

    parser.add_argument('--model', type=str, choices=['svc-rbf', 'svc-linear', 'svc-poly'],
                        default='svc-rbf', help='Specify the model to train. svc-rbf|linear|poly are available.')
    parser.add_argument('--useown', action='store', type=str, metavar='PATH',
                        help='Use your own dataset, please check dataset.py for formatting.')

    try:
        args = parser.parse_args()
    except argparse.ArgumentParserError as e:
        parser.error(str(e))
        exit(1)

    model_name = args.model

    if args.useown:
        if not os.path.isdir(args.useown):
            print(f"Error: '{args.useown}' is not a valid directory")
            exit(1)
        else:
            print('Loading data ......')
            trainimg_folder = os.path.join(args.useown, 'train')
            testimg_folder = os.path.join(args.useown, 'test')
            train_x, train_y, test_x, test_y= loadImgData(trainimg_folder, testimg_folder)
            saveData('.\\data', train_x, train_y, test_x, test_y)
            print('Successfully load data ......')
            print(f'Data are stored in: .\\data')
    else:
        data_folder = '.\\data'
        print(f'Load existed data in {data_folder} ......')
        train_x, train_y, test_x, test_y = loadData(data_folder)

    print(f'Now you are training model {model_name} ......')
    model = train(train_x, train_y, model_name)
    print(f'Now you are saving the model ......')
    abstr = model_name + '-' + str(uuid.uuid4().hex)[:6] + '.pkl'
    model_path = os.path.join('.\\model', abstr)
    saveModel(model, model_path)

    print(f'Successfully save the trained model:')
    print(f'Model is stored in: {model_path}')




if __name__ == '__main__':
    main()