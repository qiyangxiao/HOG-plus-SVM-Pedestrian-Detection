from dataset import loadData
from eval import loadModel, makeReport


modelpath = '.\\model\\svc-poly-897f38.pkl'
data_folder = '.\\data'

train_x, train_y, test_x, y_true = loadData(data_folder)
model = loadModel(modelpath)

y_pred = model.predict(test_x)
makeReport(y_true, y_pred)



