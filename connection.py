# -*- coding: utf-8 -*-
import socket
import os
import time
#import base64
from PIL import Image
import StringIO


'''
def java_byte_array_to_binary(data):
    """
    Converts a java byte array to a binary stream
    :param java byte array as string (pass in as a file like object, can use StringIO)
    :return binary string
    """
    decoded_data = base64.b64decode(data)
    strg = ''
    for i in xrange((len(decoded_data)/40)+1):
        strg += decoded_data[i*40:(i+1)*40]

    return strg
'''

def net_conn():
    HOST = '210.41.97.46'
    PORT = 1100
    
    
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    s.bind((HOST,PORT))
    s.listen(1)
    
    while True:
        total_len = 0
        total_data = []
        string = ''
        conn,addr = s.accept()
        print 'Connected by',addr
        while True:
            data = conn.recv(1024)
            if data[len(data)-1] == 't' and data[len(data)-2] == 'q':
                data = data[:-2]
                total_len += len(data)
                total_data.append(data)
                break
            total_len += len(data)
            total_data.append(data)
            
        for i in xrange(len(total_data)):
            string += total_data[i]
        image = Image.open(StringIO.StringIO(string))
        if image.format == "JPEG":
            f = open('data/1/pic.jpg', "wb")
            f.write(string)
            f.close()
            os.system('python predict.py')
            f = open('tmp.txt',"r")
            string = f.read(1024)
            f.close()
            print string
            i = conn.send(string + "\n")
                
            
    s.close()        


    
if __name__=='__main__':
    net_conn()
    
