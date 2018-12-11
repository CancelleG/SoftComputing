import numpy as np
import cx_Oracle as co
import csv

# data_test = csv.reader(open('D:\Users\HZ.Guo\PycharmProjects\SoftComputing\data.txt'))
data_test = open("D://Users//HZ.Guo//PycharmProjects//SoftComputing//test.csv")
data_test2 = open(r'D:\Users\HZ.Guo\PycharmProjects\SoftComputing\data.txt')
print(data_test.read())
print(data_test2.read())



localhost = "127.0.0.1"
port = "1521"
sid = "orcl"
dsn = co.makedsn(localhost, port, sid)
connection = co.connect('C##WEEK','C##WEEK', dsn)
##########读取数据
db0 = connection
c0 = db0.cursor()
c0.execute(r"select COLUMN_NAME,DATA_TYPE,DATA_LENGTH from user_tab_cols where table_name='PATH_GBDT_SCORE'")
row = c0.fetchone()
print(row)
#########写数据




print("ok")