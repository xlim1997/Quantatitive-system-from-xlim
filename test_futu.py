from ib_insync import IB, Stock

HOST = "127.0.0.1"
PORT = 4002   # IB Gateway Paper: 4002；TWS Paper: 7497
CLIENT_ID = 17

ib = IB()
ib.connect(HOST, PORT, clientId=CLIENT_ID)  # 只拿数据先用 readonly=True

print("Connected:", ib.isConnected())
print("Server time:", ib.reqCurrentTime())

c = Stock("AAPL", "SMART", "USD")
ib.qualifyContracts(c)

bars = ib.reqHistoricalData(
    c,
    endDateTime="",
    durationStr="30 D",
    barSizeSetting="1 day",
    whatToShow="TRADES",
    useRTH=True,
    formatDate=1
)

print("Bars:", len(bars))
if bars:
    print("Last bar:", bars[-1])

ib.disconnect()
