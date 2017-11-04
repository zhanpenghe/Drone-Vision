from preprocess import loadData, preprocess_y
from buildNetwork import buildNetwork, saveVisualizedModel
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib 

def save_model(model, y_scaler, model_path):
	model.save_weights(model_path+'/weights.h5')
	yaml_str = model.to_yaml()
	with open(model_path+'/model.txt', 'w') as f:
		f.write(yaml_str)
	joblib.dump(y_scaler, model_path+'/y_scaler.pkl')
	saveVisualizedModel(model, model_path+'/model.png')


def run():
	data_path = '../data/trainingdata/'
	print('[INFO] Start loading training data from path:\t'+data_path)
	X, y = loadData(data_path)
	print('[INFO] Finish loading all data.\n[INFO] X dimension:\t'+str(X.shape)+'\n[INFO] y dimension:\t'+str(y.shape))
	print('[INFO] Spliting dataset to train and test set.')
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
	print('[INFO] Spliting finish: X_train'+str(X_train.shape)+', X_test'+str(X_test.shape)+', y_train'+str(y_train.shape)+', y_test'+str(y_test.shape))
	print('[INFO] Preprocessing training data')
	y_train, y_test, scaler = preprocess_y(y_train, y_test)
	print('[INFO] Finish preprocessing')
	print('[INFO] Retrieving Model.')
	input_shape = (X.shape[1], X.shape[2], X.shape[3])
	output_shape = y.shape[1]
	nn = buildNetwork(input_shape, output_shape)
	nn.compile(optimizer = 'adam', loss = 'mean_absolute_error')
	print('[INFO] Finish building neural network. Start training now.')
	nn.fit(X_train, y_train, validation_data=(X_test,y_test), batch_size=10, epochs=100)
	save_model(nn, scaler, '../model')

run()