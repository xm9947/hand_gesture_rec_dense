# -*- coding: utf-8 -*-
import socket
import os



def process():    
    HOST = '127.0.0.1'
    PORT = 50001
    
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    s.bind((HOST,PORT))
    s.listen(1)
    
    while True:
        conn,addr = s.accept()
        print 'Connected by',addr
        while True:
            try:
                conn.setblocking(True)
                data = conn.recv(1024)
            except:
                print 'error!'
                pass
            print 'xxxxxxxxxx'
            if data == 'predict':
                os.system('python predict.py')
                conn.send('ok')
            else:
                conn.send('error!')
                break
        
    
    
    
        
if __name__=='__main__':
    process()        