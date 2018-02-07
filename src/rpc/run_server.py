"""
    Keras backend has to be Theano in order to run async server
"""


import msgpackrpc
import threading
import numpy as np
from keras.models import load_model


class MsgHandler(object):

    def __init__(self):

        # load model
        self.model = load_model('./models/hires_model.h5')
        self.model.load_weights('./models/hires_w1.h5')

        self.input_shape = self.model.input_shape

    def feed_forward(self, msg):
        msg = np.array(msg)
        if self._check_dim(msg):
            try:
                result = self.model.predict(msg)
                result = np.argmax(result)
            except Exception as e:
                print(e)
                result = 'err'
        else:
            result = 'wrong dimension..'

        return result

    def _check_dim(self, arr2: np.ndarray):
        print(self.input_shape, arr2.shape)
        if len(self.input_shape) != len(arr2.shape):
            return False

        for i in range(1, len(self.input_shape)):
            if self.input_shape[i] != arr2.shape[i]:
                return False

        return True


def run_server(daemon: bool=False):

    server = msgpackrpc.Server(MsgHandler())
    server.listen(msgpackrpc.Address('localhost', 8888))

    def _start_server(s):
        s.start()
        s.close()

    print(server)

    thread = threading.Thread(target=_start_server, args=(server,))
    thread.setDaemon(daemon)
    thread.start()
    return server, thread


if __name__ == '__main__':
    run_server()
