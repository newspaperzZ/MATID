import pandas as pd
from matplotlib import rcParams

from Connect import *
import matplotlib.pyplot as plt
from mpl_finance import candlestick2_ochl
from matplotlib.ticker import MultipleLocator
from matplotlib.font_manager import FontProperties

def min30_data(stockID):
    data = pd.read_sql("select stockID,everydate,startP,highP,lowP,endP from min30ZS where stockid=" + stockID + "and everydate>'2021-07-15' and everydate<'2021-08-15' order by everydate",con=conn)
    if len(data)>0:
        return data
    else:return 0

def basedata(stockID):
    data=pd.read_sql("select stockID,everydate,startP,highP,lowP,endP from dayZS where stockid="+stockID+"and everydate>'2021-06-22' and everydate<'2021-08-15' order by everydate",con=conn)
    if len(data)>0:
        return data
    else:return 0

if __name__=='__main__':
    basedata =basedata('000001')
    basedata["MA10"]=basedata["endP"].rolling(window=10).mean()
    basedata["MA20"]=basedata["endP"].rolling(window=20).mean()
    basedata1=basedata.dropna(axis=0,subset = ["MA10", "MA20"])
    print(basedata1.reset_index(drop=True))

    min30_data=min30_data('000001')
    min30_data["MA10"]=min30_data["endP"].rolling(window=10).mean()
    min30_data["MA20"]=min30_data["endP"].rolling(window=20).mean()
    min30_data1 = min30_data.dropna(axis=0, subset=["MA10", "MA20"])
    print(min30_data1.reset_index(drop=True))

    # basedata.to_csv("E:/小论文代码/自己实验/上证指数/上证指数基本数据.csv", encoding="utf-8")
    # min30_data.to_csv("E:/小论文代码/自己实验/上证指数/上证指数30min数据.csv", encoding="utf-8")

    font1 = FontProperties(fname=r"c:\windows\fonts\STZHONGS.TTF")      #华文中宋
    font2 = FontProperties(fname=r"c:\windows\fonts\timesi.ttf")    #Times New Roman
    config = {
        "font.family": 'serif',
        "font.size": 10,
        "mathtext.fontset": 'stix',
        "font.serif": ['Times New Roman'],
    }
    rcParams.update(config)     #设置全文默认字体


    # 设置窗口大小
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.xticks(range(len(basedata1["everydate"].values)), basedata1["everydate"].values, rotation=30)

    candlestick2_ochl(ax=ax,opens = basedata1["startP"].values, closes = basedata1["endP"].values, highs = basedata1["highP"].values, lows = basedata1["lowP"].values, width = 0.75, colorup = 'red', colordown = 'green')

    ax.plot(range(len(basedata1["everydate"].values)),basedata1["MA10"],color="red",label='MA10')
    ax.plot(range(len(basedata1["everydate"].values)),basedata1["MA20"],color="blue",linestyle='--',label='MA20')
    plt.legend(loc='best')
    ax.grid(True)
    plt.title("上证指数日级别趋势图",fontproperties=font1)
    plt.setp(plt.gca().get_xticklabels(), rotation=30)
    ax.set_xlabel("日期/d", loc='right', fontsize=12,fontproperties=font1)
    ax.set_ylabel('指数', loc='top', fontsize=12,fontproperties=font1)
    plt.savefig('E:\桌面\目前工作\上证指数日级别趋势图1.svg', dpi=800)
    plt.show()


    fig, ax = plt.subplots(figsize=(10, 8))
    plt.xticks(range(len(min30_data1["everydate"].values)), min30_data1["everydate"].values, rotation=30)
    xmajorLocator = MultipleLocator(5)  # 将x轴主刻度设置为5的倍数
    ax.xaxis.set_major_locator(xmajorLocator)
    # 调用方法，绘制K线图
    candlestick2_ochl(ax=ax, opens=min30_data1["startP"].values, closes=min30_data1["endP"].values,
                      highs=min30_data1["highP"].values, lows=min30_data1["lowP"].values, width=0.75, colorup='red',
                      colordown='green')
    # 如下是绘制3种均线
    ax.plot(range(len(min30_data1["everydate"].values)), min30_data1["MA10"], color="red",label='MA10')
    ax.plot(range(len(min30_data1["everydate"].values)), min30_data1["MA20"], color="blue",linestyle='--',label='MA20')
    plt.legend(loc='best')  # 绘制图例
    ax.grid(True)  # 不带网格线
    plt.title("上证指数小时级别趋势图",fontproperties=font1)
    plt.setp(plt.gca().get_xticklabels(), rotation=30)
    ax.set_xlabel("日期/d", loc='right', fontsize=12,fontproperties=font1)
    ax.set_ylabel('指数', loc='top', fontsize=12,fontproperties=font1)
    plt.savefig('E:\桌面\目前工作\上证指数小时级别趋势图1.svg', dpi=800)
    plt.show()