import hid

gamepad = hid.device()
gamepad.open(0x054c, 0x05c4)
gamepad.set_nonblocking(True)

last = None
while True:
    report = gamepad.read(64)
    if report:
        print("__________________________________________")
        for i in range(len(report)):
            if last is None:
                last = [0] * len(report)
            if report[i] != last[i] and i > 23:
                print("(" + str(i) + ", " + str(report[i]) + ")")
        last = report

