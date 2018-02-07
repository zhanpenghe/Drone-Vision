import msgpackrpc


client = msgpackrpc.Client(msgpackrpc.Address('localhost', 8888))

test_list = []


for i in range(128):
    temp = []
    for j in range(128):
        temp1 = []
        for k in range(3):
            temp1.append(i*(j-k))
        temp.append(temp1)
    test_list.append(temp)

result = client.call('feed_forward', [test_list])



print(result)
