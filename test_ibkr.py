from ib_insync import IB
import xml.etree.ElementTree as ET

ib = IB()
ib.connect("127.0.0.1", 7497, clientId=1)

xml_text = ib.reqScannerParameters()  # XML string
root = ET.fromstring(xml_text)

def get_text(node, *names):
    for n in names:
        child = node.find(n)
        if child is not None and child.text:
            return child.text.strip()
    return None

scan_types = []
for st in root.findall(".//ScanType") + root.findall(".//scanType") + root.findall(".//ScanTypeList/*"):
    scan_code = get_text(st, "scanCode", "ScanCode") or st.get("scanCode")
    instr = get_text(st, "instrument", "Instrument") or st.get("instrument")
    loc = get_text(st, "locationCode", "LocationCode") or st.get("locationCode")
    display = get_text(st, "displayName", "DisplayName") or st.get("displayName")
    if scan_code:
        scan_types.append((scan_code, instr, loc, display))

# 去重
uniq = {}
for code, instr, loc, disp in scan_types:
    uniq[(code, instr, loc)] = disp

# 例：只看美股 + 主要交易所
target_instr = "STK"          # 也可能在XML里写成 STK.US 等，看你拉到的为准
target_loc = "STK.US.MAJOR"   # 示例：美股主要交易所
filtered = [(k[0], k[1], k[2], v) for k, v in uniq.items()
            if (k[1] == target_instr or k[1] is None) and (k[2] == target_loc or k[2] is None)]

print("Total scan types:", len(uniq))
print("Filtered sample:", filtered[:30])
